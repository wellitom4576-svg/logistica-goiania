"""
Logística Profissional - Roteirização em Goiânia/GO
App com cadastro de motoristas, regiões coloridas por polígono,
atribuição inteligente de pontos e roteirização otimizada.
"""

import streamlit as st
import folium
import json
import pandas as pd
import numpy as np
from streamlit_folium import st_folium
from folium.plugins import Draw, MarkerCluster
from shapely.geometry import shape, Point, Polygon, MultiPolygon, MultiPoint
from shapely.ops import unary_union
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
import branca.colormap as cm
import io
import zipfile
import colorsys
import os
import math

# ── Config ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Logística Goiânia",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded",
)

GOIANIA_CENTER = [-16.6869, -49.2648]
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# 10 cores fixas e bem distintas para motoristas (uma por motorista)
CORES_MOTORISTAS = [
    "#e74c3c",  # Vermelho
    "#2ecc71",  # Verde
    "#3498db",  # Azul
    "#f39c12",  # Laranja
    "#9b59b6",  # Roxo
    "#1abc9c",  # Turquesa
    "#e67e22",  # Laranja escuro
    "#e84393",  # Rosa
    "#00b894",  # Menta
    "#6c5ce7",  # Violeta
]

NOMES_CORES = [
    "Vermelho", "Verde", "Azul", "Laranja", "Roxo",
    "Turquesa", "Laranja Esc.", "Rosa", "Menta", "Violeta",
]

ICONES_MOTORISTAS = [
    "🔴", "🟢", "🔵", "🟠", "🟣", "🩵", "🟧", "🩷", "🟩", "🟪",
]

FOLIUM_COLORS = [
    "red", "green", "blue", "orange", "purple",
    "cadetblue", "darkred", "pink", "darkgreen", "darkpurple",
]


# ── Session State ───────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "pontos_coleta": [],
        "rota_otimizada": None,
        "poligonos": [],
        "base_location": GOIANIA_CENTER,
        "motoristas": [],
        "atribuicao_motorista": {},  # {idx_ponto: idx_motorista}
        "regioes_motoristas": {},    # {idx_motorista: [(lat,lon),...]}
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── Funções Utilitárias ────────────────────────────────────────────────────
def gerar_cores(n):
    cores = []
    for i in range(n):
        hue = i / max(n, 1)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
        cores.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")
    return cores


def calcular_matriz_distancias(pontos):
    n = len(pontos)
    matriz = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = geodesic(
                (pontos[i]["lat"], pontos[i]["lon"]),
                (pontos[j]["lat"], pontos[j]["lon"]),
            ).meters
            matriz[i][j] = int(dist)
            matriz[j][i] = int(dist)
    return matriz


# ═══════════════════════════════════════════════════════════════════════════
# CLUSTERIZAÇÃO PROFISSIONAL
# ═══════════════════════════════════════════════════════════════════════════

def _dist_geo(lat1, lon1, lat2, lon2):
    """Distância geodésica em metros (cache-friendly)."""
    return geodesic((lat1, lon1), (lat2, lon2)).meters


def _kmeans_geo(pontos, k, base_lat, base_lon, max_iter=80):
    """K-Means geodésico com inicialização por varredura angular."""
    n = len(pontos)
    coords = np.array([[p["lat"], p["lon"]] for p in pontos])

    angulos = [math.atan2(p["lat"] - base_lat, p["lon"] - base_lon) for p in pontos]
    idx_sorted = sorted(range(n), key=lambda i: angulos[i])
    step = max(n // k, 1)
    centroids = np.array([coords[idx_sorted[i * step % n]] for i in range(k)])

    labels = np.zeros(n, dtype=int)

    for _ in range(max_iter):
        for i in range(n):
            best_d = float("inf")
            for c in range(k):
                d = _dist_geo(coords[i][0], coords[i][1],
                              centroids[c][0], centroids[c][1])
                if d < best_d:
                    best_d = d
                    labels[i] = c

        new_centroids = np.copy(centroids)
        for c in range(k):
            mask = labels == c
            if mask.any():
                new_centroids[c] = coords[mask].mean(axis=0)

        if np.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids

    return labels


def agrupar_pontos_por_regiao(pontos, num_motoristas, base_lat, base_lon):
    """
    Clusterização profissional em 6 passos:
    1. K-Means geodésico → blocos iniciais por proximidade
    2. Detecção de outliers → pontos >1.8x média migram para cluster mais próximo
    3. Verificação de fronteira → pontos na borda entre 2 clusters vão para o mais compacto
    4. Balanceamento → diferença max 2 pontos entre clusters
    5. Ordenação angular → clusters em sentido horário a partir da base
    6. Validação de compacidade → nenhum cluster deve ter raio > 1.5x o menor raio
    """
    if not pontos or num_motoristas < 1:
        return {}
    n = len(pontos)
    if num_motoristas >= n:
        return {i: [i] for i in range(n)}

    coords = np.array([[p["lat"], p["lon"]] for p in pontos])

    # ── Passo 1: K-Means geodésico
    labels = _kmeans_geo(pontos, num_motoristas, base_lat, base_lon)

    # ── Passo 2: Outliers — mover pontos > 1.8x média do cluster
    for _pass in range(5):
        moved = False
        for c in range(num_motoristas):
            idx_c = np.where(labels == c)[0]
            if len(idx_c) < 3:
                continue
            centroid = coords[idx_c].mean(axis=0)
            dists = np.array([_dist_geo(coords[i][0], coords[i][1],
                                        centroid[0], centroid[1])
                              for i in idx_c])
            threshold = dists.mean() * 1.8

            for j, i in enumerate(idx_c):
                if dists[j] <= threshold:
                    continue
                # Achar cluster mais próximo
                best_c, best_d = c, dists[j]
                for c2 in range(num_motoristas):
                    if c2 == c:
                        continue
                    idx_c2 = np.where(labels == c2)[0]
                    if len(idx_c2) == 0:
                        continue
                    cent2 = coords[idx_c2].mean(axis=0)
                    d2 = _dist_geo(coords[i][0], coords[i][1], cent2[0], cent2[1])
                    if d2 < best_d:
                        best_d = d2
                        best_c = c2
                if best_c != c:
                    labels[i] = best_c
                    moved = True
        if not moved:
            break

    # ── Passo 3: Fronteira — pontos equidistantes entre 2 clusters
    # Para cada ponto, se a diferença de distância aos 2 clusters mais
    # próximos é < 20%, mover para o cluster que fica mais compacto
    for i in range(n):
        c_atual = labels[i]
        dists_to_clusters = []
        for c in range(num_motoristas):
            idx_c = np.where(labels == c)[0]
            if len(idx_c) == 0:
                dists_to_clusters.append(float("inf"))
                continue
            cent = coords[idx_c].mean(axis=0)
            dists_to_clusters.append(
                _dist_geo(coords[i][0], coords[i][1], cent[0], cent[1]))

        sorted_c = sorted(range(num_motoristas), key=lambda c: dists_to_clusters[c])
        c1, c2 = sorted_c[0], sorted_c[1]
        d1, d2 = dists_to_clusters[c1], dists_to_clusters[c2]

        if d1 > 0 and (d2 - d1) / d1 < 0.20 and c_atual != c1:
            # Checar qual cluster fica mais compacto com este ponto
            def _raio_com(cluster_id, ponto_idx, incluir):
                idx = list(np.where(labels == cluster_id)[0])
                if incluir and ponto_idx not in idx:
                    idx.append(ponto_idx)
                elif not incluir and ponto_idx in idx:
                    idx.remove(ponto_idx)
                if len(idx) < 2:
                    return 0
                cent = coords[idx].mean(axis=0)
                return max(_dist_geo(coords[j][0], coords[j][1],
                                     cent[0], cent[1]) for j in idx)

            raio_c1_com = _raio_com(c1, i, True)
            raio_atual_sem = _raio_com(c_atual, i, False)
            raio_c1_sem = _raio_com(c1, i, False)
            raio_atual_com = _raio_com(c_atual, i, True)

            if raio_c1_com + raio_atual_sem < raio_c1_sem + raio_atual_com:
                labels[i] = c1

    # ── Passo 4: Balanceamento (diferença máxima = 2 pontos)
    for _pass in range(n * 2):
        sizes = [int((labels == c).sum()) for c in range(num_motoristas)]
        max_c = max(range(num_motoristas), key=lambda c: sizes[c])
        min_c = min(range(num_motoristas), key=lambda c: sizes[c])
        if sizes[max_c] - sizes[min_c] <= 2:
            break

        idx_max = np.where(labels == max_c)[0]
        mask_min = labels == min_c
        cent_min = coords[mask_min].mean(axis=0) if mask_min.any() else coords.mean(axis=0)

        # Mover o ponto do cluster grande que está mais perto do cluster pequeno
        # MAS que não é central para seu cluster atual
        cent_max = coords[idx_max].mean(axis=0)
        dists_to_own = np.array([
            _dist_geo(coords[i][0], coords[i][1], cent_max[0], cent_max[1])
            for i in idx_max
        ])
        dists_to_target = np.array([
            _dist_geo(coords[i][0], coords[i][1], cent_min[0], cent_min[1])
            for i in idx_max
        ])

        # Score = distância ao destino - distância ao próprio (pontos na periferia
        # do cluster grande e perto do pequeno são os melhores candidatos)
        scores = dists_to_target - dists_to_own * 0.5
        move_i = idx_max[np.argmin(scores)]
        labels[move_i] = min_c

    # ── Passo 5: Ordenar clusters por ângulo (sentido horário)
    cluster_angles = []
    for c in range(num_motoristas):
        mask = labels == c
        if mask.any():
            cent = coords[mask].mean(axis=0)
            angle = math.atan2(cent[0] - base_lat, cent[1] - base_lon)
        else:
            angle = 999
        cluster_angles.append((c, angle))

    cluster_angles.sort(key=lambda x: x[1])
    remap = {old_c: new_c for new_c, (old_c, _) in enumerate(cluster_angles)}

    # ── Passo 6: Validação de compacidade
    # Se um cluster tem raio > 2x o raio do menor cluster, tentar dividir
    raios = {}
    for c in range(num_motoristas):
        idx_c = np.where(labels == c)[0]
        if len(idx_c) < 2:
            raios[c] = 0
            continue
        cent = coords[idx_c].mean(axis=0)
        raios[c] = max(_dist_geo(coords[i][0], coords[i][1],
                                  cent[0], cent[1]) for i in idx_c)

    raios_validos = [r for r in raios.values() if r > 0]
    if raios_validos:
        min_raio = min(raios_validos)
        for c in range(num_motoristas):
            if raios[c] > min_raio * 2.5:
                # Mover o ponto mais distante para o cluster mais próximo
                idx_c = np.where(labels == c)[0]
                if len(idx_c) <= 3:
                    continue
                cent = coords[idx_c].mean(axis=0)
                dists = [_dist_geo(coords[i][0], coords[i][1],
                                   cent[0], cent[1]) for i in idx_c]
                outlier_i = idx_c[np.argmax(dists)]
                # Achar melhor cluster vizinho
                best_c, best_d = c, max(dists)
                for c2 in range(num_motoristas):
                    if c2 == c:
                        continue
                    idx_c2 = np.where(labels == c2)[0]
                    if len(idx_c2) == 0:
                        continue
                    cent2 = coords[idx_c2].mean(axis=0)
                    d2 = _dist_geo(coords[outlier_i][0], coords[outlier_i][1],
                                   cent2[0], cent2[1])
                    if d2 < best_d:
                        best_d = d2
                        best_c = c2
                if best_c != c:
                    labels[outlier_i] = best_c

    # ── Montar resultado
    grupos = {c: [] for c in range(num_motoristas)}
    for i in range(n):
        new_c = remap[labels[i]]
        grupos[new_c].append(i)

    return grupos


# ═══════════════════════════════════════════════════════════════════════════
# PÓS-PROCESSAMENTO: 2-OPT + VARREDURA ANGULAR
# ═══════════════════════════════════════════════════════════════════════════

def _segmentos_cruzam(p1, p2, p3, p4):
    """Verifica se segmento (p1→p2) cruza (p3→p4) usando produto vetorial."""
    def ccw(A, B, C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
    return (ccw(p1,p3,p4) != ccw(p2,p3,p4)) and (ccw(p1,p2,p3) != ccw(p1,p2,p4))


def _two_opt(rota, matriz):
    """
    2-opt: elimina cruzamentos invertendo sub-sequências.
    Roda até não haver mais melhorias.
    """
    n = len(rota)
    if n < 4:
        return rota

    melhorou = True
    while melhorou:
        melhorou = False
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                # Custo atual: rota[i-1]→rota[i] + rota[j]→rota[j+1]
                # Custo novo:  rota[i-1]→rota[j] + rota[i]→rota[j+1]
                custo_atual = (matriz[rota[i-1]][rota[i]] +
                               matriz[rota[j]][rota[j+1] if j+1 < n else rota[0]])
                custo_novo = (matriz[rota[i-1]][rota[j]] +
                              matriz[rota[i]][rota[j+1] if j+1 < n else rota[0]])

                if custo_novo < custo_atual - 1:  # margem de 1m
                    rota[i:j+1] = rota[i:j+1][::-1]
                    melhorou = True
    return rota


def _or_opt(rota, matriz):
    """
    Or-opt: move blocos de 1, 2 ou 3 pontos para melhor posição.
    Complementa o 2-opt para sequências mais naturais.
    """
    n = len(rota)
    if n < 4:
        return rota

    for block_size in [1, 2, 3]:
        melhorou = True
        while melhorou:
            melhorou = False
            for i in range(1, n - block_size):
                bloco = rota[i:i + block_size]
                if i + block_size >= n:
                    continue

                # Custo de remover o bloco
                antes = rota[i - 1]
                depois = rota[i + block_size] if i + block_size < n else rota[0]
                custo_remover = (
                    matriz[antes][bloco[0]] +
                    matriz[bloco[-1]][depois] -
                    matriz[antes][depois]
                )

                # Testar inserir em cada outra posição
                for j in range(1, n):
                    if abs(j - i) <= block_size:
                        continue
                    if j >= n:
                        continue

                    prev_j = rota[j - 1] if j > 0 else rota[-1]
                    next_j = rota[j] if j < n else rota[0]

                    custo_inserir = (
                        matriz[prev_j][bloco[0]] +
                        matriz[bloco[-1]][next_j] -
                        matriz[prev_j][next_j]
                    )

                    if custo_inserir < custo_remover - 10:
                        # Mover bloco
                        nova_rota = rota[:i] + rota[i + block_size:]
                        # Ajustar índice de inserção
                        ins = j if j < i else j - block_size
                        ins = max(1, min(ins, len(nova_rota)))
                        nova_rota = nova_rota[:ins] + bloco + nova_rota[ins:]
                        rota = nova_rota
                        n = len(rota)
                        melhorou = True
                        break
                if melhorou:
                    break
    return rota


def _varredura_angular(pontos_indices, pontos, deposito_lat, deposito_lon):
    """
    Ordena pontos por varredura angular a partir do depósito.
    Cria uma sequência natural de progressão territorial.
    """
    if len(pontos_indices) <= 1:
        return pontos_indices

    def angulo(idx):
        return math.atan2(
            pontos[idx]["lat"] - deposito_lat,
            pontos[idx]["lon"] - deposito_lon,
        )

    return sorted(pontos_indices, key=angulo)


def criar_convex_hull(pontos_coords):
    if len(pontos_coords) < 3:
        return None
    try:
        points = MultiPoint([(lon, lat) for lat, lon in pontos_coords])
        hull = points.convex_hull
        if hull.geom_type == "Polygon":
            return hull
    except Exception:
        pass
    return None


# ═══════════════════════════════════════════════════════════════════════════
# OTIMIZAÇÃO DE ROTA PROFISSIONAL
# ═══════════════════════════════════════════════════════════════════════════

def otimizar_rota(pontos, num_veiculos=1, deposito=0, max_dist_km=300,
                  balancear=True, preco_litro=6.29, km_por_litro=10.0,
                  tempo_busca_seg=15):
    """
    Otimização profissional de rotas em 3 fases:

    FASE 1 — OR-Tools VRP com penalidade de saltos longos
      Custo = distância + penalidade cúbica para trechos > mediana
      3 estratégias em paralelo, seleciona a melhor

    FASE 2 — 2-opt: elimina todos os cruzamentos de rota
      Inverte sub-sequências sempre que reduz distância

    FASE 3 — Or-opt: move blocos de 1-3 pontos para posição ótima
      Ajusta a sequência final para progressão territorial natural
    """
    if len(pontos) < 2:
        return None

    matriz = calcular_matriz_distancias(pontos)

    # Limiar adaptativo de penalidade
    dists_flat = matriz[matriz > 0]
    mediana_dist = float(np.median(dists_flat)) if len(dists_flat) > 0 else 5000
    p75 = float(np.percentile(dists_flat, 75)) if len(dists_flat) > 0 else 7000
    limiar = mediana_dist * 1.2  # mais agressivo: penaliza a partir de 1.2x mediana

    manager = pywrapcp.RoutingIndexManager(len(pontos), num_veiculos, deposito)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        dist = int(matriz[from_node][to_node])

        # Penalidade cúbica para saltos longos (muito mais agressiva)
        if dist > limiar:
            excesso = (dist - limiar) / limiar
            penalidade = int(dist * excesso * excesso)
            return dist + penalidade
        return dist

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Dimensão real (sem penalidade) para limite de km
    def distance_real_callback(from_index, to_index):
        return int(matriz[manager.IndexToNode(from_index)]
                         [manager.IndexToNode(to_index)])

    real_transit_idx = routing.RegisterTransitCallback(distance_real_callback)
    routing.AddDimension(
        real_transit_idx, 0, int(max_dist_km * 1000), True, "RealDistance",
    )
    if balancear and num_veiculos > 1:
        routing.GetDimensionOrDie("RealDistance").SetGlobalSpanCostCoefficient(150)

    # ── FASE 1: Multi-strategy OR-Tools
    best_solution = None
    best_cost = float("inf")

    strategies = [
        (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
         routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH),
        (routing_enums_pb2.FirstSolutionStrategy.SAVINGS,
         routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH),
        (routing_enums_pb2.FirstSolutionStrategy.CHRISTOFIDES,
         routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING),
    ]

    time_per = max(tempo_busca_seg // len(strategies), 5)

    for first_strat, meta in strategies:
        sp = pywrapcp.DefaultRoutingSearchParameters()
        sp.first_solution_strategy = first_strat
        sp.local_search_metaheuristic = meta
        sp.time_limit.FromSeconds(time_per)
        sp.log_search = False
        solution = routing.SolveWithParameters(sp)
        if solution and solution.ObjectiveValue() < best_cost:
            best_cost = solution.ObjectiveValue()
            best_solution = solution

    solution = best_solution
    if not solution:
        return None

    rotas = []
    for vehicle_id in range(num_veiculos):
        # ── Extrair nós do OR-Tools
        rota_nodes = []
        index = routing.Start(vehicle_id)
        while not routing.IsEnd(index):
            rota_nodes.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        rota_nodes.append(manager.IndexToNode(index))  # depósito final

        # Pular rotas vazias (só depósito → depósito)
        pontos_internos = [n for n in rota_nodes if n != deposito]
        if not pontos_internos:
            continue

        # ── FASE 2 + 3: Pós-processamento nos pontos internos
        if len(pontos_internos) >= 3:
            # Montar sub-rota sem depósitos para otimizar
            sub = [deposito] + pontos_internos + [deposito]
            sub = _two_opt(sub, matriz)   # elimina cruzamentos
            sub = _or_opt(sub, matriz)    # move blocos para posição ótima
            # Reconstruir rota completa garantindo depósito nas pontas
            rota_nodes = sub if sub[0] == deposito else [deposito] + sub
            if rota_nodes[-1] != deposito:
                rota_nodes.append(deposito)

        # ── Recalcular trechos com a rota pós-processada
        trechos = []
        distancia_total = 0
        for i in range(len(rota_nodes) - 1):
            node = rota_nodes[i]
            next_node = rota_nodes[i + 1]
            dist_trecho = matriz[node][next_node]
            distancia_total += dist_trecho
            trechos.append({
                "de_idx": node, "para_idx": next_node,
                "de_nome": pontos[node].get("nome", f"Ponto {node}"),
                "para_nome": pontos[next_node].get("nome", f"Ponto {next_node}"),
                "de_lat": pontos[node]["lat"], "de_lon": pontos[node]["lon"],
                "para_lat": pontos[next_node]["lat"], "para_lon": pontos[next_node]["lon"],
                "distancia_m": round(dist_trecho),
                "distancia_km": round(dist_trecho / 1000, 2),
            })

        paradas = []
        acumulado = 0.0
        for seq, node_idx in enumerate(rota_nodes):
            if seq > 0:
                acumulado += trechos[seq - 1]["distancia_km"]
            paradas.append({
                "ordem": seq, "node_idx": node_idx,
                "nome": pontos[node_idx].get("nome", f"Ponto {node_idx}"),
                "lat": pontos[node_idx]["lat"], "lon": pontos[node_idx]["lon"],
                "endereco": pontos[node_idx].get("endereco", ""),
                "obs": pontos[node_idx].get("obs", ""),
                "dist_trecho_km": trechos[seq - 1]["distancia_km"] if seq > 0 else 0.0,
                "dist_acumulada_km": round(acumulado, 2),
            })

        num_coletas = len([n for n in rota_nodes if n != deposito]) - (
            1 if rota_nodes[-1] == deposito else 0
        )

        dist_km = round(distancia_total / 1000, 2)
        litros = round(dist_km / km_por_litro, 2)
        custo = round(litros * preco_litro, 2)

        rotas.append({
            "rota": rota_nodes, "trechos": trechos, "paradas": paradas,
            "distancia_km": dist_km,
            "num_coletas": num_coletas,
            "litros_estimados": litros,
            "custo_combustivel": custo,
            "km_por_litro": km_por_litro,
            "preco_litro": preco_litro,
            "motorista": f"Motorista {vehicle_id + 1}",
        })

    return rotas


def geocodificar(endereco):
    geolocator = Nominatim(user_agent="logistica_goiania_app")
    try:
        location = geolocator.geocode(f"{endereco}, Goiânia, GO, Brasil")
        if location:
            return {"lat": location.latitude, "lon": location.longitude}
    except Exception:
        pass
    return None


def carregar_geojson_bairros(arquivo):
    try:
        content = arquivo.read().decode("utf-8")
        return json.loads(content)
    except Exception as e:
        st.error(f"Erro ao carregar GeoJSON: {e}")
        return None


def ponto_no_poligono(lat, lon, geojson):
    ponto = Point(lon, lat)
    for feature in geojson.get("features", []):
        try:
            geom = shape(feature["geometry"])
            if geom.contains(ponto):
                nome = feature.get("properties", {}).get(
                    "name", feature.get("properties", {}).get(
                        "NOME", feature.get("properties", {}).get("nome", "Sem nome")))
                return nome
        except Exception:
            continue
    return "Fora dos limites"


def criar_mapa(pontos, rota=None, geojson_bairros=None, poligonos_custom=None,
               motoristas=None, atribuicao=None):
    """Cria mapa com regiões coloridas por motorista."""
    m = folium.Map(location=GOIANIA_CENTER, zoom_start=12, tiles="CartoDB positron")

    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Satélite",
    ).add_to(m)

    # ── Polígonos de bairros (GeoJSON importado)
    if geojson_bairros:
        fg_bairros = folium.FeatureGroup(name="Bairros", show=True)
        cores_bairros = gerar_cores(len(geojson_bairros.get("features", [])))
        for i, feature in enumerate(geojson_bairros.get("features", [])):
            cor = cores_bairros[i % len(cores_bairros)]
            nome = feature.get("properties", {}).get(
                "name", feature.get("properties", {}).get(
                    "NOME", feature.get("properties", {}).get("nome", f"Bairro {i+1}")))
            folium.GeoJson(
                feature, name=nome,
                style_function=lambda x, c=cor: {
                    "fillColor": c, "color": c, "weight": 2, "fillOpacity": 0.10,
                },
                tooltip=nome,
            ).add_to(fg_bairros)
        fg_bairros.add_to(m)

    # ── Polígonos de regiões dos motoristas (convex hull colorido)
    if motoristas and atribuicao and len(pontos) > 1:
        fg_regioes = folium.FeatureGroup(name="Regioes Motoristas", show=True)

        # Agrupar pontos por motorista
        pontos_por_mot = {}
        for p_idx, m_idx in atribuicao.items():
            if m_idx not in pontos_por_mot:
                pontos_por_mot[m_idx] = []
            # p_idx é índice nos pontos_coleta, +1 para pular a base
            real_idx = int(p_idx) + 1
            if real_idx < len(pontos):
                pontos_por_mot[m_idx].append((pontos[real_idx]["lat"], pontos[real_idx]["lon"]))

        for m_idx, coords in pontos_por_mot.items():
            if m_idx >= len(motoristas):
                continue
            cor = CORES_MOTORISTAS[m_idx % len(CORES_MOTORISTAS)]
            nome_mot = motoristas[m_idx]["nome"]

            if len(coords) >= 3:
                hull = criar_convex_hull(coords)
                if hull:
                    # Expandir o hull um pouco para ficar mais visível
                    hull_expanded = hull.buffer(0.003)  # ~300m
                    coords_hull = list(hull_expanded.exterior.coords)
                    folium.Polygon(
                        locations=[[lat, lon] for lon, lat in coords_hull],
                        color=cor, weight=3, fill=True,
                        fill_color=cor, fill_opacity=0.15,
                        tooltip=f"Regiao: {nome_mot}",
                        popup=f"<b>{nome_mot}</b><br>{len(coords)} pontos",
                    ).add_to(fg_regioes)
            elif len(coords) == 2:
                folium.PolyLine(
                    locations=coords, color=cor, weight=4, opacity=0.6,
                    dash_array="10", tooltip=f"Regiao: {nome_mot}",
                ).add_to(fg_regioes)
            # Para 1 ponto, apenas o marcador com cor já é suficiente

        fg_regioes.add_to(m)

    # ── Polígonos customizados (desenhados no mapa)
    if poligonos_custom:
        for poly in poligonos_custom:
            folium.Polygon(
                locations=poly["coords"], color=poly.get("cor", "#ff7800"),
                weight=2, fill=True, fill_color=poly.get("cor", "#ff7800"),
                fill_opacity=0.2, tooltip=poly.get("nome", "Regiao"),
            ).add_to(m)

    # ── Base / Depósito
    if pontos:
        folium.Marker(
            location=[pontos[0]["lat"], pontos[0]["lon"]],
            popup=f"<b>BASE:</b> {pontos[0].get('nome', 'Depósito')}",
            icon=folium.Icon(color="red", icon="home", prefix="fa"),
            tooltip="Base / Depósito",
        ).add_to(m)

    # ── Pontos de coleta com cor do motorista atribuído
    for i, p in enumerate(pontos[1:], 1):
        p_idx = i - 1  # índice no pontos_coleta
        mot_idx = atribuicao.get(p_idx, None) if atribuicao else None

        if mot_idx is not None and motoristas and mot_idx < len(motoristas):
            cor_folium = FOLIUM_COLORS[mot_idx % len(FOLIUM_COLORS)]
            cor_hex = CORES_MOTORISTAS[mot_idx % len(CORES_MOTORISTAS)]
            nome_mot = motoristas[mot_idx]["nome"]
            motorista_info = f"<br><span style='color:{cor_hex};font-weight:bold'>■ {nome_mot}</span>"
        else:
            cor_folium = "gray"
            motorista_info = "<br><i>Sem motorista</i>"

        bairro_info = ""
        if geojson_bairros:
            bairro = ponto_no_poligono(p["lat"], p["lon"], geojson_bairros)
            bairro_info = f"<br>Bairro: {bairro}"

        popup_html = f"""
        <div style="min-width:220px">
            <b>#{i} - {p.get('nome', f'Ponto {i}')}</b>
            {motorista_info}
            <br>Lat: {p['lat']:.6f} | Lon: {p['lon']:.6f}
            {bairro_info}
            {f"<br>End: {p.get('endereco', '')}" if p.get('endereco') else ""}
            {f"<br>Obs: {p.get('obs', '')}" if p.get('obs') else ""}
        </div>
        """
        folium.Marker(
            location=[p["lat"], p["lon"]],
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.Icon(color=cor_folium, icon="box", prefix="fa"),
            tooltip=f"#{i} - {p.get('nome', f'Ponto {i}')}",
        ).add_to(m)

    # ── Rota otimizada
    if rota:
        for v_idx, r in enumerate(rota):
            cor = CORES_MOTORISTAS[v_idx % len(CORES_MOTORISTAS)]
            coords_rota = []
            for node_idx in r["rota"]:
                coords_rota.append([pontos[node_idx]["lat"], pontos[node_idx]["lon"]])

            folium.PolyLine(
                coords_rota, color=cor, weight=5, opacity=0.85,
                tooltip=f"{r.get('motorista', f'Veiculo {v_idx+1}')} - {r['distancia_km']} km",
            ).add_to(m)

            for seq, node_idx in enumerate(r["rota"]):
                if seq == 0:
                    continue
                folium.CircleMarker(
                    location=[pontos[node_idx]["lat"], pontos[node_idx]["lon"]],
                    radius=14, color=cor, fill=True, fill_color=cor,
                    fill_opacity=0.9, tooltip=f"Parada {seq}",
                ).add_to(m)
                folium.Marker(
                    location=[pontos[node_idx]["lat"], pontos[node_idx]["lon"]],
                    icon=folium.DivIcon(
                        html=f'<div style="font-size:11px;color:white;font-weight:bold;'
                             f'text-align:center;line-height:28px">{seq}</div>',
                        icon_size=(28, 28), icon_anchor=(14, 14),
                    ),
                ).add_to(m)

    Draw(
        draw_options={
            "polyline": False, "rectangle": True, "polygon": True,
            "circle": False, "marker": True, "circlemarker": False,
        },
        edit_options={"edit": True, "remove": True},
    ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


# ══════════════════════════════════════════════════════════════════════════════
# CSS PROFISSIONAL
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Header principal ───────────────────────────────────────────────── */
.header-pro {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 28px 32px;
    border-radius: 16px;
    margin-bottom: 24px;
    box-shadow: 0 8px 32px rgba(0,0,0,.18);
    position: relative;
    overflow: hidden;
}
.header-pro::before {
    content: "";
    position: absolute;
    top: -40%; right: -10%;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(52,152,219,.15) 0%, transparent 70%);
    border-radius: 50%;
}
.header-pro h1 {
    color: #fff;
    margin: 0 0 4px 0;
    font-size: 1.85rem;
    font-weight: 700;
    letter-spacing: -.5px;
}
.header-pro p {
    color: rgba(255,255,255,.65);
    margin: 0;
    font-size: .95rem;
}

/* ── Cards de KPI ───────────────────────────────────────────────────── */
.kpi-row {
    display: flex;
    gap: 14px;
    flex-wrap: wrap;
    margin: 16px 0;
}
.kpi-card {
    flex: 1 1 160px;
    background: #fff;
    border: 1px solid #e8ecf1;
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,.04);
    transition: transform .15s, box-shadow .15s;
}
.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,0,0,.08);
}
.kpi-card .kpi-icon {
    font-size: 1.6rem;
    margin-bottom: 4px;
}
.kpi-card .kpi-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #1a1a2e;
    line-height: 1.2;
}
.kpi-card .kpi-label {
    font-size: .78rem;
    color: #7f8c8d;
    text-transform: uppercase;
    letter-spacing: .5px;
    margin-top: 2px;
}
.kpi-card.green  { border-top: 3px solid #2ecc71; }
.kpi-card.blue   { border-top: 3px solid #3498db; }
.kpi-card.orange { border-top: 3px solid #f39c12; }
.kpi-card.red    { border-top: 3px solid #e74c3c; }
.kpi-card.purple { border-top: 3px solid #9b59b6; }

/* ── Motorista card ─────────────────────────────────────────────────── */
.mot-card {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 14px 18px;
    background: #fff;
    border: 1px solid #e8ecf1;
    border-radius: 10px;
    margin-bottom: 8px;
    transition: box-shadow .15s;
}
.mot-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,.07); }
.mot-badge {
    width: 42px; height: 42px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #fff;
    font-weight: 700;
    font-size: 1.1rem;
    flex-shrink: 0;
}
.mot-info { flex: 1; }
.mot-info .mot-name { font-weight: 600; font-size: 1rem; color: #1a1a2e; }
.mot-info .mot-detail { font-size: .82rem; color: #7f8c8d; }
.mot-pontos {
    background: #f0f2f6;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: .82rem;
    font-weight: 600;
    color: #2c3e50;
}

/* ── Seção ──────────────────────────────────────────────────────────── */
.section-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #1a1a2e;
    margin: 24px 0 10px 0;
    padding-bottom: 8px;
    border-bottom: 2px solid #3498db;
    display: inline-block;
}
.section-subtitle {
    font-size: .92rem;
    color: #7f8c8d;
    margin: -6px 0 14px 0;
}

/* ── Legenda mapa ───────────────────────────────────────────────────── */
.map-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    padding: 10px 0;
}
.map-legend-item {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    background: #f8f9fb;
    border: 1px solid #e8ecf1;
    border-radius: 20px;
    font-size: .82rem;
    font-weight: 500;
}
.map-legend-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
}

/* ── Status bar ─────────────────────────────────────────────────────── */
.status-bar {
    display: flex;
    gap: 20px;
    padding: 10px 18px;
    background: #f8f9fb;
    border: 1px solid #e8ecf1;
    border-radius: 8px;
    margin: 8px 0 16px 0;
    font-size: .85rem;
    color: #2c3e50;
}
.status-bar b { color: #1a1a2e; }

/* ── Economia banner ────────────────────────────────────────────────── */
.eco-banner {
    background: linear-gradient(135deg, #d4efdf 0%, #a9dfbf 100%);
    border: 1px solid #82e0aa;
    border-radius: 12px;
    padding: 16px 22px;
    margin: 12px 0;
}
.eco-banner .eco-title {
    font-weight: 700;
    color: #1e8449;
    font-size: 1rem;
    margin-bottom: 6px;
}
.eco-banner .eco-values {
    display: flex;
    gap: 24px;
    flex-wrap: wrap;
}
.eco-banner .eco-item {
    font-size: .88rem;
    color: #1a5032;
}
.eco-banner .eco-item b {
    font-size: 1.05rem;
}

/* ── Rota card ──────────────────────────────────────────────────────── */
.rota-header-card {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 14px 18px;
    background: #fff;
    border: 1px solid #e8ecf1;
    border-left: 4px solid;
    border-radius: 10px;
    margin-bottom: 8px;
}
.rota-header-card .rota-icon {
    font-size: 1.8rem;
}
.rota-header-card .rota-info { flex: 1; }
.rota-header-card .rota-name { font-weight: 700; font-size: 1.05rem; color: #1a1a2e; }
.rota-header-card .rota-stats {
    font-size: .82rem;
    color: #7f8c8d;
    margin-top: 2px;
}

/* ── Trajeto visual ─────────────────────────────────────────────────── */
.trajeto-line {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 4px;
    padding: 10px 0;
    font-size: .82rem;
}
.trajeto-stop {
    background: #eaf2f8;
    padding: 3px 10px;
    border-radius: 6px;
    font-weight: 600;
    color: #2c3e50;
    white-space: nowrap;
}
.trajeto-arrow {
    color: #bdc3c7;
    font-size: .7rem;
}
.trajeto-dist {
    color: #7f8c8d;
    font-size: .72rem;
    font-style: italic;
}

/* ── Footer ─────────────────────────────────────────────────────────── */
.footer-pro {
    text-align: center;
    padding: 18px;
    margin-top: 30px;
    border-top: 1px solid #e8ecf1;
    color: #95a5a6;
    font-size: .78rem;
}

/* ── Sidebar styling ────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #f8f9fb;
}
section[data-testid="stSidebar"] .stMarkdown h2 {
    color: #1a1a2e;
    font-size: 1.1rem;
    border-bottom: 2px solid #3498db;
    padding-bottom: 6px;
}

/* ── Tab styling ────────────────────────────────────────────────────── */
button[data-baseweb="tab"] {
    font-weight: 600 !important;
    font-size: .88rem !important;
}

/* ── Esconder menu hamburguer e "Made with" ─────────────────────────── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HEADER PROFISSIONAL
# ══════════════════════════════════════════════════════════════════════════════
n_pts = len(st.session_state.pontos_coleta)
n_mot = len(st.session_state.motoristas)
n_poly = len(st.session_state.poligonos)
tem_rota = st.session_state.rota_otimizada is not None

st.markdown(f"""
<div class="header-pro">
    <h1>Logistica Profissional</h1>
    <p>Roteirizacao inteligente &bull; Goiania/GO &bull;
       {n_pts} pontos &bull; {n_mot} motoristas
       {"&bull; Rota otimizada" if tem_rota else ""}</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR PROFISSIONAL
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## Configuracoes")

    # ── Base
    with st.expander("Base (Ponto de Partida)", expanded=True):
        base_endereco = st.text_input("Endereco da base", value="Centro, Goiania, GO")
        sb1, sb2 = st.columns(2)
        with sb1:
            base_lat = st.number_input("Latitude", value=-16.6869, format="%.6f", key="blat")
        with sb2:
            base_lon = st.number_input("Longitude", value=-49.2648, format="%.6f", key="blon")

        if st.button("Geocodificar Base", use_container_width=True):
            loc = geocodificar(base_endereco)
            if loc:
                st.session_state.base_location = [loc["lat"], loc["lon"]]
                st.success(f"Base: {loc['lat']:.6f}, {loc['lon']:.6f}")
            else:
                st.error("Endereco nao encontrado")

    # ── Importar pontos
    with st.expander("Importar Pontos de Coleta", expanded=not bool(n_pts)):
        arquivo_pontos = st.file_uploader(
            "CSV, XLSX ou JSON", type=["csv", "xlsx", "xls", "json"],
            key="upload_pontos", label_visibility="collapsed",
        )

        if arquivo_pontos:
            try:
                if arquivo_pontos.name.endswith(".csv"):
                    df = pd.read_csv(arquivo_pontos)
                elif arquivo_pontos.name.endswith((".xlsx", ".xls")):
                    df = pd.read_excel(arquivo_pontos)
                else:
                    df = pd.DataFrame(json.loads(arquivo_pontos.read().decode("utf-8")))

                col_map = {}
                for col in df.columns:
                    cl = col.lower().strip()
                    if cl in ("latitude", "lat"):
                        col_map[col] = "lat"
                    elif cl in ("longitude", "lon", "lng", "long"):
                        col_map[col] = "lon"
                    elif cl in ("nome", "name", "local"):
                        col_map[col] = "nome"
                    elif cl in ("endereco", "endereço", "address"):
                        col_map[col] = "endereco"
                    elif cl in ("obs", "observacao", "observação", "nota"):
                        col_map[col] = "obs"
                df = df.rename(columns=col_map)

                if "lat" in df.columns and "lon" in df.columns:
                    pontos_importados = []
                    for _, row in df.iterrows():
                        pontos_importados.append({
                            "lat": float(row["lat"]),
                            "lon": float(row["lon"]),
                            "nome": str(row.get("nome", "")),
                            "endereco": str(row.get("endereco", "")),
                            "obs": str(row.get("obs", "")),
                        })
                    st.success(f"{len(pontos_importados)} pontos importados!")
                    st.session_state.pontos_coleta = pontos_importados
                    st.session_state.atribuicao_motorista = {}
                else:
                    st.error("Colunas 'lat' e 'lon' nao encontradas.")
            except Exception as e:
                st.error(f"Erro: {e}")

    # ── GeoJSON bairros
    with st.expander("Poligonos de Bairros"):
        arquivo_geojson = st.file_uploader(
            "Arquivo GeoJSON", type=["geojson", "json"], key="upload_geojson",
            label_visibility="collapsed",
        )
        geojson_bairros = None
        if arquivo_geojson:
            geojson_bairros = carregar_geojson_bairros(arquivo_geojson)
            if geojson_bairros:
                st.success(f"{len(geojson_bairros.get('features', []))} poligonos!")

        if st.button("Baixar Bairros (OSM)", use_container_width=True):
            with st.spinner("Baixando..."):
                try:
                    import osmnx as ox
                    gdf = ox.features_from_place(
                        "Goiania, Goias, Brazil",
                        tags={"admin_level": "10", "boundary": "administrative"},
                    )
                    if len(gdf) == 0:
                        gdf = ox.features_from_place(
                            "Goiania, Goias, Brazil",
                            tags={"place": ["neighbourhood", "suburb"]},
                        )
                    if len(gdf) > 0:
                        gdf_poly = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]
                        if len(gdf_poly) > 0:
                            geojson_str = gdf_poly.to_json()
                            geojson_bairros = json.loads(geojson_str)
                            st.success(f"{len(gdf_poly)} bairros!")
                            st.download_button("Salvar GeoJSON", geojson_str,
                                               "bairros_goiania.geojson", "application/json")
                except Exception as e:
                    st.error(f"Erro: {e}")

    # ── Combustivel
    with st.expander("Combustivel e Veiculo"):
        preco_litro = st.number_input(
            "Preco (R$/L)", min_value=1.0, max_value=15.0,
            value=6.29, step=0.10, format="%.2f", key="preco_litro_sb",
        )
        km_por_litro = st.number_input(
            "Consumo (km/L)", min_value=3.0, max_value=30.0,
            value=10.0, step=0.5, format="%.1f", key="km_litro_sb",
        )
        max_dist = st.number_input(
            "Dist. maxima/veiculo (km)",
            min_value=10, max_value=1000, value=300, step=10, key="max_dist_sb",
        )
        tempo_busca = st.select_slider(
            "Nivel de otimizacao",
            options=[10, 20, 30, 60],
            value=20,
            format_func=lambda x: {
                10: "Rapido", 20: "Normal",
                30: "Profundo", 60: "Maximo"
            }[x],
            key="tempo_sb",
        )


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD KPI (topo, antes das abas)
# ══════════════════════════════════════════════════════════════════════════════
rotas_ativas = st.session_state.rota_otimizada
if rotas_ativas:
    _dist_t = sum(r["distancia_km"] for r in rotas_ativas)
    _lit_t = sum(r.get("litros_estimados", 0) for r in rotas_ativas)
    _cust_t = sum(r.get("custo_combustivel", 0) for r in rotas_ativas)
    _col_t = sum(r["num_coletas"] for r in rotas_ativas)
    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi-card blue">
            <div class="kpi-icon">📍</div>
            <div class="kpi-value">{n_pts}</div>
            <div class="kpi-label">Pontos de Coleta</div>
        </div>
        <div class="kpi-card green">
            <div class="kpi-icon">👷</div>
            <div class="kpi-value">{len(rotas_ativas)}</div>
            <div class="kpi-label">Motoristas Ativos</div>
        </div>
        <div class="kpi-card orange">
            <div class="kpi-icon">🛣️</div>
            <div class="kpi-value">{_dist_t:.1f} km</div>
            <div class="kpi-label">Distancia Total</div>
        </div>
        <div class="kpi-card red">
            <div class="kpi-icon">⛽</div>
            <div class="kpi-value">{_lit_t:.1f} L</div>
            <div class="kpi-label">Combustivel</div>
        </div>
        <div class="kpi-card purple">
            <div class="kpi-icon">💰</div>
            <div class="kpi-value">R$ {_cust_t:.2f}</div>
            <div class="kpi-label">Custo Total</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi-card blue">
            <div class="kpi-icon">📍</div>
            <div class="kpi-value">{n_pts}</div>
            <div class="kpi-label">Pontos de Coleta</div>
        </div>
        <div class="kpi-card green">
            <div class="kpi-icon">👷</div>
            <div class="kpi-value">{n_mot}</div>
            <div class="kpi-label">Motoristas</div>
        </div>
        <div class="kpi-card orange">
            <div class="kpi-icon">🔷</div>
            <div class="kpi-value">{n_poly}</div>
            <div class="kpi-label">Poligonos</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ABAS PRINCIPAIS
# ══════════════════════════════════════════════════════════════════════════════
tab_mapa, tab_rota, tab_motoristas, tab_pontos, tab_relatorio = st.tabs([
    "Mapa", "Roteirizacao", "Motoristas",
    "Pontos de Coleta", "Relatorio",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB MAPA
# ══════════════════════════════════════════════════════════════════════════════
with tab_mapa:
    todos_pontos = [
        {"lat": base_lat, "lon": base_lon, "nome": "BASE / Deposito"}
    ] + st.session_state.pontos_coleta

    mapa = criar_mapa(
        todos_pontos,
        rota=st.session_state.rota_otimizada,
        geojson_bairros=geojson_bairros,
        poligonos_custom=st.session_state.poligonos,
        motoristas=st.session_state.motoristas,
        atribuicao=st.session_state.atribuicao_motorista,
    )

    map_data = st_folium(mapa, width=None, height=620, key="main_map")

    # Capturar desenhos
    if map_data and map_data.get("all_drawings"):
        drawings = map_data["all_drawings"]
        new_polys = []
        new_points = []
        for d in drawings:
            geom = d.get("geometry", {})
            if geom.get("type") == "Polygon":
                coords = geom["coordinates"][0]
                new_polys.append({
                    "coords": [[c[1], c[0]] for c in coords],
                    "cor": "#ff7800",
                    "nome": f"Regiao desenhada {len(new_polys)+1}",
                })
            elif geom.get("type") == "Point":
                lon, lat = geom["coordinates"]
                new_points.append({"lat": lat, "lon": lon, "nome": "Ponto manual"})

        if new_polys:
            st.session_state.poligonos = new_polys
        if new_points:
            for p in new_points:
                if p not in st.session_state.pontos_coleta:
                    st.session_state.pontos_coleta.append(p)

    # Legenda profissional
    if st.session_state.motoristas:
        leg_items = ""
        for i, mot in enumerate(st.session_state.motoristas):
            cor = mot["cor"]
            pontos_mot = sum(
                1 for v in st.session_state.atribuicao_motorista.values() if v == i
            )
            leg_items += (
                f'<div class="map-legend-item">'
                f'<div class="map-legend-dot" style="background:{cor}"></div>'
                f'<span>{mot["nome"]} ({pontos_mot})</span>'
                f'</div>'
            )
        st.markdown(f'<div class="map-legend">{leg_items}</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="status-bar">'
        f'<span>📍 <b>{n_pts}</b> pontos</span>'
        f'<span>👷 <b>{n_mot}</b> motoristas</span>'
        f'<span>🔷 <b>{n_poly}</b> poligonos</span>'
        f'{"<span>✅ <b>Rota otimizada</b></span>" if tem_rota else ""}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB ROTEIRIZACAO
# ══════════════════════════════════════════════════════════════════════════════
with tab_rota:
    st.markdown('<div class="section-title">Roteirizacao Profissional</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">OR-Tools + 2-opt + Or-opt com penalidade cubica</div>',
                unsafe_allow_html=True)

    if len(st.session_state.pontos_coleta) < 2:
        st.warning("Adicione pelo menos 2 pontos para otimizar.")
    elif not st.session_state.motoristas:
        st.warning("Cadastre motoristas na aba 'Motoristas' primeiro.")
    else:
        num_mot = len(st.session_state.motoristas)

        # ── Resumo pre-otimizacao
        st.markdown(f"""
        <div class="kpi-row">
            <div class="kpi-card blue">
                <div class="kpi-value">{n_pts}</div>
                <div class="kpi-label">Pontos</div>
            </div>
            <div class="kpi-card green">
                <div class="kpi-value">{num_mot}</div>
                <div class="kpi-label">Motoristas</div>
            </div>
            <div class="kpi-card orange">
                <div class="kpi-value">{n_pts / max(num_mot,1):.1f}</div>
                <div class="kpi-label">Media pts/mot</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_modo, col_bal = st.columns([3, 1])
        with col_modo:
            modo_rota = st.radio(
                "Modo de roteirizacao",
                ["Por motorista (usa atribuicao)", "Otimizacao global (OR-Tools divide)"],
                horizontal=True,
            )
        with col_bal:
            balancear = st.checkbox("Balancear carga", value=True)

        if st.button("OTIMIZAR ROTAS", type="primary", use_container_width=True):
            with st.spinner("Calculando rotas otimizadas... Aguarde."):

                if modo_rota == "Por motorista (usa atribuicao)":
                    # Auto-distribuir se nao houver atribuicao
                    if not st.session_state.atribuicao_motorista:
                        n_m = len(st.session_state.motoristas)
                        pontos = st.session_state.pontos_coleta
                        grupos = agrupar_pontos_por_regiao(pontos, n_m, base_lat, base_lon)
                        nova_atrib = {}
                        for m_idx, indices in grupos.items():
                            for p_idx in indices:
                                nova_atrib[p_idx] = m_idx
                        st.session_state.atribuicao_motorista = nova_atrib
                        st.info(f"Pontos distribuidos automaticamente entre {n_m} motoristas")

                    if st.session_state.atribuicao_motorista:
                        todas_rotas = []
                        base = {"lat": base_lat, "lon": base_lon, "nome": "BASE / Deposito"}

                        for m_idx, mot in enumerate(st.session_state.motoristas):
                            pontos_idx = [
                                k for k, v in st.session_state.atribuicao_motorista.items()
                                if v == m_idx
                            ]
                            if not pontos_idx:
                                continue

                            pontos_mot = [base] + [
                                st.session_state.pontos_coleta[i] for i in pontos_idx
                            ]

                            resultado = otimizar_rota(
                                pontos_mot, 1, deposito=0,
                                max_dist_km=max_dist, balancear=False,
                                preco_litro=preco_litro,
                                km_por_litro=km_por_litro,
                                tempo_busca_seg=tempo_busca,
                            )

                            if resultado:
                                r = resultado[0]
                                r["motorista"] = mot["nome"]
                                r["motorista_idx"] = m_idx
                                r["cor"] = mot["cor"]
                                todas_rotas.append(r)

                        if todas_rotas:
                            todos_pontos_global = [base] + st.session_state.pontos_coleta
                            rotas_globais = []

                            for rota_mot in todas_rotas:
                                m_idx = rota_mot["motorista_idx"]
                                pontos_idx = sorted([
                                    k for k, v in st.session_state.atribuicao_motorista.items()
                                    if v == m_idx
                                ])

                                mapa_local_global = {0: 0}
                                for local_i, global_i in enumerate(pontos_idx, 1):
                                    mapa_local_global[local_i] = global_i + 1

                                nova_rota = [
                                    mapa_local_global.get(nd, nd) for nd in rota_mot["rota"]
                                ]
                                novas_paradas = []
                                for p in rota_mot["paradas"]:
                                    p_copy = dict(p)
                                    p_copy["node_idx"] = mapa_local_global.get(
                                        p["node_idx"], p["node_idx"]
                                    )
                                    novas_paradas.append(p_copy)

                                rotas_globais.append({
                                    **rota_mot,
                                    "rota": nova_rota,
                                    "paradas": novas_paradas,
                                })

                            st.session_state.rota_otimizada = rotas_globais
                            st.session_state.todos_pontos_rota = todos_pontos_global
                            st.rerun()
                        else:
                            st.error("Nenhuma rota encontrada.")

                else:
                    # Otimizacao global
                    todos_pontos = [
                        {"lat": base_lat, "lon": base_lon, "nome": "BASE / Deposito"}
                    ] + st.session_state.pontos_coleta

                    resultado = otimizar_rota(
                        todos_pontos, num_mot, deposito=0,
                        max_dist_km=max_dist, balancear=balancear,
                        preco_litro=preco_litro,
                        km_por_litro=km_por_litro,
                        tempo_busca_seg=tempo_busca,
                    )

                    if resultado:
                        for i, r in enumerate(resultado):
                            if i < len(st.session_state.motoristas):
                                r["motorista"] = st.session_state.motoristas[i]["nome"]
                                r["motorista_idx"] = i
                                r["cor"] = st.session_state.motoristas[i]["cor"]

                        st.session_state.rota_otimizada = resultado
                        st.session_state.todos_pontos_rota = todos_pontos

                        nova_atribuicao = {}
                        for r_idx, r in enumerate(resultado):
                            m_idx = r.get("motorista_idx", r_idx)
                            for node in r["rota"]:
                                if node > 0:
                                    nova_atribuicao[node - 1] = m_idx
                        st.session_state.atribuicao_motorista = nova_atribuicao
                        st.rerun()
                    else:
                        st.error("Sem solucao. Aumente distancia maxima ou motoristas.")

        # ── RESULTADOS DA OTIMIZACAO ──────────────────────────────────────
        if st.session_state.rota_otimizada:
            rotas = st.session_state.rota_otimizada
            todos_pontos = st.session_state.get("todos_pontos_rota", [])

            dist_total = sum(r["distancia_km"] for r in rotas)
            coletas_total = sum(r["num_coletas"] for r in rotas)
            litros_total = sum(r.get("litros_estimados", 0) for r in rotas)
            custo_total = sum(r.get("custo_combustivel", 0) for r in rotas)

            # ── Banner de economia
            if todos_pontos and len(todos_pontos) > 2:
                dist_sequencial = 0
                for i in range(len(todos_pontos) - 1):
                    dist_sequencial += geodesic(
                        (todos_pontos[i]["lat"], todos_pontos[i]["lon"]),
                        (todos_pontos[i+1]["lat"], todos_pontos[i+1]["lon"]),
                    ).km
                dist_sequencial += geodesic(
                    (todos_pontos[-1]["lat"], todos_pontos[-1]["lon"]),
                    (todos_pontos[0]["lat"], todos_pontos[0]["lon"]),
                ).km

                if dist_sequencial > dist_total:
                    eco_km = dist_sequencial - dist_total
                    eco_pct = (eco_km / dist_sequencial) * 100
                    eco_lit = eco_km / km_por_litro
                    eco_rs = eco_lit * preco_litro

                    st.markdown(f"""
                    <div class="eco-banner">
                        <div class="eco-title">Economia com Otimizacao</div>
                        <div class="eco-values">
                            <div class="eco-item"><b>{eco_km:.1f} km</b> a menos ({eco_pct:.0f}%)</div>
                            <div class="eco-item"><b>{eco_lit:.1f} litros</b> economizados</div>
                            <div class="eco-item"><b>R$ {eco_rs:.2f}</b> de economia</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # ── KPIs resultado
            custo_por_col = custo_total / max(coletas_total, 1)
            st.markdown(f"""
            <div class="kpi-row">
                <div class="kpi-card blue">
                    <div class="kpi-value">{len(rotas)}</div>
                    <div class="kpi-label">Motoristas Ativos</div>
                </div>
                <div class="kpi-card green">
                    <div class="kpi-value">{coletas_total}</div>
                    <div class="kpi-label">Total Coletas</div>
                </div>
                <div class="kpi-card orange">
                    <div class="kpi-value">{dist_total:.1f} km</div>
                    <div class="kpi-label">Distancia Total</div>
                </div>
                <div class="kpi-card red">
                    <div class="kpi-value">{litros_total:.1f} L</div>
                    <div class="kpi-label">Combustivel Total</div>
                </div>
                <div class="kpi-card purple">
                    <div class="kpi-value">R$ {custo_por_col:.2f}</div>
                    <div class="kpi-label">Custo / Coleta</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Tabela comparativa
            st.markdown('<div class="section-title">Divisao por Motorista</div>',
                        unsafe_allow_html=True)

            df_comp = pd.DataFrame([{
                "Motorista": r["motorista"],
                "Coletas": r["num_coletas"],
                "Distancia (km)": r["distancia_km"],
                "Litros": r.get("litros_estimados", 0),
                "Custo (R$)": r.get("custo_combustivel", 0),
                "% Dist": round(r["distancia_km"] / max(dist_total, 0.01) * 100, 1),
                "% Coletas": round(r["num_coletas"] / max(coletas_total, 1) * 100, 1),
            } for r in rotas])

            st.dataframe(df_comp, use_container_width=True, hide_index=True,
                         column_config={
                             "Distancia (km)": st.column_config.NumberColumn(format="%.2f"),
                             "Litros": st.column_config.NumberColumn(format="%.1f"),
                             "Custo (R$)": st.column_config.NumberColumn(format="R$ %.2f"),
                             "% Dist": st.column_config.ProgressColumn(
                                 min_value=0, max_value=100, format="%.1f%%"),
                             "% Coletas": st.column_config.ProgressColumn(
                                 min_value=0, max_value=100, format="%.1f%%"),
                         })

            ch1, ch2 = st.columns(2)
            with ch1:
                st.bar_chart(df_comp.set_index("Motorista")["Distancia (km)"], color="#e74c3c")
            with ch2:
                st.bar_chart(df_comp.set_index("Motorista")["Coletas"], color="#3498db")

            # ── Roteiro detalhado
            st.markdown('<div class="section-title">Roteiro Detalhado</div>',
                        unsafe_allow_html=True)

            for v_idx, r in enumerate(rotas):
                m_idx = r.get("motorista_idx", v_idx)
                icone = ICONES_MOTORISTAS[m_idx % len(ICONES_MOTORISTAS)]
                cor = CORES_MOTORISTAS[m_idx % len(CORES_MOTORISTAS)]
                maior_trecho = max(
                    (t["distancia_km"] for t in r["trechos"]), default=0
                ) if r["trechos"] else 0

                # Card de header para cada motorista
                st.markdown(f"""
                <div class="rota-header-card" style="border-left-color: {cor}">
                    <div class="rota-icon">{icone}</div>
                    <div class="rota-info">
                        <div class="rota-name">{r['motorista']}</div>
                        <div class="rota-stats">
                            {r['num_coletas']} coletas &bull;
                            {r['distancia_km']} km &bull;
                            {r.get('litros_estimados', 0)} L &bull;
                            R$ {r.get('custo_combustivel', 0):.2f} &bull;
                            Maior trecho: {maior_trecho} km
                        </div>
                    </div>
                    <div class="mot-pontos">{r['num_coletas']} pts</div>
                </div>
                """, unsafe_allow_html=True)

                with st.expander(f"Ver detalhes - {r['motorista']}", expanded=(v_idx == 0)):
                    # Trajeto visual
                    traj_html = '<div class="trajeto-line">'
                    for p_i, p in enumerate(r["paradas"]):
                        nome_curto = p["nome"][:25]
                        if p_i > 0:
                            dist_t = r["trechos"][p_i-1]["distancia_km"]
                            traj_html += f'<span class="trajeto-dist">{dist_t}km</span>'
                            traj_html += '<span class="trajeto-arrow">&#9654;</span>'
                        traj_html += f'<span class="trajeto-stop">{nome_curto}</span>'
                    traj_html += '</div>'
                    st.markdown(traj_html, unsafe_allow_html=True)

                    # Tabela de paradas
                    df_paradas = pd.DataFrame([{
                        "#": p["ordem"],
                        "Local": p["nome"],
                        "Endereco": p["endereco"],
                        "Trecho (km)": p["dist_trecho_km"] if p["ordem"] > 0 else "-",
                        "Acumulado (km)": p["dist_acumulada_km"],
                    } for p in r["paradas"]])
                    st.dataframe(df_paradas, use_container_width=True, hide_index=True)

            # ── Exportacao
            st.markdown('<div class="section-title">Exportar Roteiros</div>',
                        unsafe_allow_html=True)
            exp1, exp2, exp3 = st.columns(3)

            with exp1:
                all_rows = []
                for r in rotas:
                    for p in r["paradas"]:
                        all_rows.append({
                            "Motorista": r["motorista"],
                            "Ordem": p["ordem"], "Local": p["nome"],
                            "Endereco": p["endereco"], "Obs": p["obs"],
                            "Lat": p["lat"], "Lon": p["lon"],
                            "Trecho (km)": p["dist_trecho_km"],
                            "Acumulado (km)": p["dist_acumulada_km"],
                        })
                st.download_button("Roteiro CSV", pd.DataFrame(all_rows).to_csv(index=False),
                                   "roteiro_completo.csv", "text/csv",
                                   use_container_width=True)

            with exp2:
                zip_buf = io.BytesIO()
                with zipfile.ZipFile(zip_buf, "w") as zf:
                    for r in rotas:
                        rows = [{
                            "Ordem": p["ordem"], "Local": p["nome"],
                            "Endereco": p["endereco"], "Obs": p["obs"],
                            "Lat": p["lat"], "Lon": p["lon"],
                            "Trecho (km)": p["dist_trecho_km"],
                            "Acumulado (km)": p["dist_acumulada_km"],
                        } for p in r["paradas"]]
                        fname = r["motorista"].replace(" ", "_").lower()
                        zf.writestr(f"{fname}.csv",
                                    pd.DataFrame(rows).to_csv(index=False))
                st.download_button("Por Motorista (ZIP)", zip_buf.getvalue(),
                                   "roteiros_motoristas.zip", "application/zip",
                                   use_container_width=True)

            with exp3:
                export_json = json.dumps([{
                    "motorista": r["motorista"],
                    "distancia_km": r["distancia_km"],
                    "num_coletas": r["num_coletas"],
                    "paradas": r["paradas"], "trechos": r["trechos"],
                } for r in rotas], ensure_ascii=False, indent=2)
                st.download_button("Roteiro JSON", export_json,
                                   "roteiro.json", "application/json",
                                   use_container_width=True)

            if st.button("Limpar Rotas", type="secondary"):
                st.session_state.rota_otimizada = None
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB MOTORISTAS
# ══════════════════════════════════════════════════════════════════════════════
with tab_motoristas:
    st.markdown('<div class="section-title">Equipe de Motoristas</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Cadastre ate 10 motoristas com cores exclusivas</div>',
                unsafe_allow_html=True)

    # ── Formulario
    with st.expander("Adicionar Novo Motorista", expanded=not bool(st.session_state.motoristas)):
        fm1, fm2, fm3 = st.columns([2, 2, 1])
        with fm1:
            novo_mot_nome = st.text_input("Nome", key="novo_mot_nome",
                                          placeholder="Ex: Joao Silva")
        with fm2:
            novo_mot_tel = st.text_input("Telefone", key="novo_mot_tel",
                                         placeholder="(62) 99999-0000")
        with fm3:
            novo_mot_placa = st.text_input("Placa", key="novo_mot_placa",
                                           placeholder="ABC-1234")

        if st.button("Adicionar Motorista", type="primary",
                     disabled=len(st.session_state.motoristas) >= 10,
                     use_container_width=True):
            if novo_mot_nome.strip():
                idx = len(st.session_state.motoristas)
                st.session_state.motoristas.append({
                    "nome": novo_mot_nome.strip(),
                    "telefone": novo_mot_tel.strip(),
                    "placa": novo_mot_placa.strip(),
                    "cor": CORES_MOTORISTAS[idx % len(CORES_MOTORISTAS)],
                    "cor_nome": NOMES_CORES[idx % len(NOMES_CORES)],
                })
                st.rerun()
            else:
                st.error("Nome obrigatorio")

    # ── Lista de motoristas cadastrados
    if not st.session_state.motoristas:
        st.info("Nenhum motorista cadastrado. Clique acima para comecar.")

    remover_idx = None
    for i in range(10):
        if i < len(st.session_state.motoristas):
            mot = st.session_state.motoristas[i]
            cor = mot["cor"]
            pontos_atrib = sum(
                1 for v in st.session_state.atribuicao_motorista.values() if v == i
            )
            st.markdown(f"""
            <div class="mot-card">
                <div class="mot-badge" style="background:{cor}">{i+1}</div>
                <div class="mot-info">
                    <div class="mot-name">{mot['nome']}</div>
                    <div class="mot-detail">
                        {mot['cor_nome']}
                        {(' | ' + mot['telefone']) if mot.get('telefone') else ''}
                        {(' | ' + mot['placa']) if mot.get('placa') else ''}
                    </div>
                </div>
                <div class="mot-pontos">{pontos_atrib} pontos</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Remover", key=f"rm_mot_{i}"):
                remover_idx = i

    if remover_idx is not None:
        st.session_state.motoristas.pop(remover_idx)
        new_atrib = {}
        for pk, mv in st.session_state.atribuicao_motorista.items():
            if mv == remover_idx:
                continue
            elif mv > remover_idx:
                new_atrib[pk] = mv - 1
            else:
                new_atrib[pk] = mv
        st.session_state.atribuicao_motorista = new_atrib
        st.rerun()

    # ── Atribuicao de pontos
    if st.session_state.motoristas and st.session_state.pontos_coleta:
        st.markdown('<div class="section-title">Atribuicao de Pontos</div>',
                    unsafe_allow_html=True)

        atrib_mode = st.radio(
            "Modo",
            ["Automatica (por regiao)", "Manual (ponto a ponto)"],
            horizontal=True, label_visibility="collapsed",
        )

        if atrib_mode == "Automatica (por regiao)":
            st.caption(
                "Distribui os pontos em regioes geograficas automaticamente. "
                "Cada motorista recebe uma fatia da cidade."
            )
            if st.button("Distribuir Automaticamente", type="primary",
                         use_container_width=True):
                n_m = len(st.session_state.motoristas)
                pontos = st.session_state.pontos_coleta
                grupos = agrupar_pontos_por_regiao(pontos, n_m, base_lat, base_lon)

                nova_atribuicao = {}
                for m_idx, indices in grupos.items():
                    for p_idx in indices:
                        nova_atribuicao[p_idx] = m_idx

                st.session_state.atribuicao_motorista = nova_atribuicao
                st.rerun()
        else:
            nomes_mot = ["-- Sem motorista --"] + [
                f"{ICONES_MOTORISTAS[i]} {m['nome']}"
                for i, m in enumerate(st.session_state.motoristas)
            ]

            nova_atribuicao = dict(st.session_state.atribuicao_motorista)
            for p_idx, p in enumerate(st.session_state.pontos_coleta):
                current = st.session_state.atribuicao_motorista.get(p_idx, None)
                default_idx = (current + 1) if current is not None else 0

                sel = st.selectbox(
                    f"#{p_idx+1} - {p.get('nome', 'Ponto')[:40]}",
                    options=range(len(nomes_mot)),
                    format_func=lambda x: nomes_mot[x],
                    index=default_idx,
                    key=f"atrib_{p_idx}",
                )
                if sel == 0:
                    nova_atribuicao.pop(p_idx, None)
                else:
                    nova_atribuicao[p_idx] = sel - 1

            if st.button("Salvar Atribuicoes", use_container_width=True):
                st.session_state.atribuicao_motorista = nova_atribuicao
                st.success("Atribuicoes salvas!")
                st.rerun()

    # ── Resumo
    if st.session_state.atribuicao_motorista and st.session_state.motoristas:
        st.markdown('<div class="section-title">Resumo da Distribuicao</div>',
                    unsafe_allow_html=True)

        resumo_data = []
        for m_idx, mot in enumerate(st.session_state.motoristas):
            pontos_idx = [
                k for k, v in st.session_state.atribuicao_motorista.items() if v == m_idx
            ]
            dist_est = 0
            if len(pontos_idx) > 1:
                coords = [
                    (st.session_state.pontos_coleta[i]["lat"],
                     st.session_state.pontos_coleta[i]["lon"])
                    for i in pontos_idx
                ]
                for j in range(len(coords) - 1):
                    dist_est += geodesic(coords[j], coords[j+1]).km

            resumo_data.append({
                "Cor": ICONES_MOTORISTAS[m_idx],
                "Motorista": mot["nome"],
                "Pontos": len(pontos_idx),
                "Dist. Est. (km)": round(dist_est, 2),
            })

        nao_atrib = len(st.session_state.pontos_coleta) - len(
            st.session_state.atribuicao_motorista
        )
        if nao_atrib > 0:
            resumo_data.append({
                "Cor": "---",
                "Motorista": "SEM MOTORISTA",
                "Pontos": nao_atrib,
                "Dist. Est. (km)": 0,
            })

        st.dataframe(pd.DataFrame(resumo_data), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB PONTOS DE COLETA
# ══════════════════════════════════════════════════════════════════════════════
with tab_pontos:
    st.markdown('<div class="section-title">Pontos de Coleta</div>',
                unsafe_allow_html=True)

    with st.expander("Adicionar Ponto Manualmente"):
        p1, p2, p3 = st.columns(3)
        with p1:
            novo_nome = st.text_input("Nome", key="novo_nome", placeholder="Nome do ponto")
        with p2:
            novo_lat = st.number_input("Latitude", value=-16.6869, format="%.6f", key="nlat")
        with p3:
            novo_lon = st.number_input("Longitude", value=-49.2648, format="%.6f", key="nlon")

        p4, p5 = st.columns(2)
        with p4:
            novo_end = st.text_input("Endereco", key="novo_end", placeholder="Opcional")
        with p5:
            nova_obs = st.text_input("Observacao", key="nova_obs", placeholder="Opcional")

        btn1, btn2 = st.columns(2)
        with btn1:
            if st.button("Adicionar Ponto", use_container_width=True):
                st.session_state.pontos_coleta.append({
                    "lat": novo_lat, "lon": novo_lon,
                    "nome": novo_nome or f"Ponto {len(st.session_state.pontos_coleta)+1}",
                    "endereco": novo_end, "obs": nova_obs,
                })
                st.rerun()
        with btn2:
            if st.button("Geocodificar e Adicionar", use_container_width=True) and novo_end:
                loc = geocodificar(novo_end)
                if loc:
                    st.session_state.pontos_coleta.append({
                        "lat": loc["lat"], "lon": loc["lon"],
                        "nome": novo_nome or novo_end,
                        "endereco": novo_end, "obs": nova_obs,
                    })
                    st.rerun()
                else:
                    st.error("Endereco nao encontrado")

    if st.session_state.pontos_coleta:
        st.markdown(
            f'<div class="status-bar">'
            f'<span>📍 <b>{len(st.session_state.pontos_coleta)}</b> pontos cadastrados</span>'
            f'<span>✅ <b>{len(st.session_state.atribuicao_motorista)}</b> atribuidos</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        pontos_display = []
        for i, p in enumerate(st.session_state.pontos_coleta):
            mot_idx = st.session_state.atribuicao_motorista.get(i, None)
            mot_nome = ""
            if mot_idx is not None and mot_idx < len(st.session_state.motoristas):
                mot_nome = st.session_state.motoristas[mot_idx]["nome"]
            pontos_display.append({**p, "motorista": mot_nome})

        df_pontos = pd.DataFrame(pontos_display)
        edited_df = st.data_editor(
            df_pontos, num_rows="dynamic", use_container_width=True, key="editor_pontos",
        )

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            if st.button("Salvar Alteracoes", use_container_width=True):
                records = edited_df.to_dict("records")
                st.session_state.pontos_coleta = [
                    {k: v for k, v in r.items() if k != "motorista"} for r in records
                ]
                st.rerun()
        with c2:
            st.download_button("Exportar CSV", df_pontos.to_csv(index=False),
                               "pontos_coleta.csv", "text/csv", use_container_width=True)
        with c3:
            st.download_button("Exportar JSON",
                               df_pontos.to_json(orient="records", force_ascii=False),
                               "pontos_coleta.json", "application/json",
                               use_container_width=True)
        with c4:
            if st.button("Limpar Pontos", type="secondary", use_container_width=True):
                st.session_state.pontos_coleta = []
                st.session_state.rota_otimizada = None
                st.session_state.atribuicao_motorista = {}
                st.rerun()
    else:
        st.info("Nenhum ponto cadastrado. Importe um arquivo na barra lateral ou adicione manualmente.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB RELATORIO
# ══════════════════════════════════════════════════════════════════════════════
with tab_relatorio:
    st.markdown('<div class="section-title">Relatorio Gerencial</div>',
                unsafe_allow_html=True)

    if st.session_state.pontos_coleta:
        todos_pontos = st.session_state.get("todos_pontos_rota", [
            {"lat": base_lat, "lon": base_lon, "nome": "BASE"}
        ] + st.session_state.pontos_coleta)

        rotas = st.session_state.rota_otimizada

        if rotas:
            dist_total = sum(r["distancia_km"] for r in rotas)
            litros_total = sum(r.get("litros_estimados", 0) for r in rotas)
            custo_total = sum(r.get("custo_combustivel", 0) for r in rotas)
            coletas_total = sum(r["num_coletas"] for r in rotas)

            st.markdown(f"""
            <div class="kpi-row">
                <div class="kpi-card blue">
                    <div class="kpi-icon">📍</div>
                    <div class="kpi-value">{n_pts}</div>
                    <div class="kpi-label">Pontos</div>
                </div>
                <div class="kpi-card green">
                    <div class="kpi-icon">👷</div>
                    <div class="kpi-value">{len(rotas)}</div>
                    <div class="kpi-label">Motoristas</div>
                </div>
                <div class="kpi-card orange">
                    <div class="kpi-icon">🛣️</div>
                    <div class="kpi-value">{dist_total:.1f} km</div>
                    <div class="kpi-label">Distancia</div>
                </div>
                <div class="kpi-card red">
                    <div class="kpi-icon">⛽</div>
                    <div class="kpi-value">{litros_total:.1f} L</div>
                    <div class="kpi-label">Combustivel</div>
                </div>
                <div class="kpi-card purple">
                    <div class="kpi-icon">💰</div>
                    <div class="kpi-value">R$ {custo_total:.2f}</div>
                    <div class="kpi-label">Custo Total</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Equilibrio
            if len(rotas) > 1:
                st.markdown('<div class="section-title">Equilibrio da Divisao</div>',
                            unsafe_allow_html=True)
                dists = [r["distancia_km"] for r in rotas]

                eq1, eq2, eq3, eq4 = st.columns(4)
                with eq1:
                    st.metric("Desvio Padrao", f"{np.std(dists):.2f} km")
                with eq2:
                    st.metric("Razao Max/Min", f"{max(dists)/max(min(dists),0.01):.1f}x")
                with eq3:
                    st.metric("Mais Distante", f"{max(dists):.2f} km")
                with eq4:
                    st.metric("Menos Distante", f"{min(dists):.2f} km")

            df_rel = pd.DataFrame([{
                "Motorista": r["motorista"],
                "Coletas": r["num_coletas"],
                "Dist (km)": r["distancia_km"],
                "Litros": r.get("litros_estimados", 0),
                "Custo (R$)": r.get("custo_combustivel", 0),
                "km/Coleta": round(r["distancia_km"] / max(r["num_coletas"], 1), 2),
                "R$/Coleta": round(r.get("custo_combustivel", 0) / max(r["num_coletas"], 1), 2),
                "Maior Trecho (km)": max(
                    (t["distancia_km"] for t in r["trechos"]), default=0),
            } for r in rotas])
            st.dataframe(df_rel, use_container_width=True, hide_index=True)

            # ── Top trechos
            st.markdown('<div class="section-title">Top 10 Maiores Trechos</div>',
                        unsafe_allow_html=True)
            trechos_info = []
            for r in rotas:
                for t in r["trechos"]:
                    trechos_info.append({
                        "Motorista": r["motorista"],
                        "De": t["de_nome"], "Para": t["para_nome"],
                        "Distancia (km)": t["distancia_km"],
                    })
            df_top = pd.DataFrame(trechos_info).sort_values(
                "Distancia (km)", ascending=False).head(10)
            st.dataframe(df_top, use_container_width=True, hide_index=True)

        else:
            st.markdown(f"""
            <div class="kpi-row">
                <div class="kpi-card blue">
                    <div class="kpi-icon">📍</div>
                    <div class="kpi-value">{n_pts}</div>
                    <div class="kpi-label">Pontos</div>
                </div>
                <div class="kpi-card green">
                    <div class="kpi-icon">👷</div>
                    <div class="kpi-value">{n_mot}</div>
                    <div class="kpi-label">Motoristas</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.info("Otimize as rotas para ver o relatorio completo.")

        # ── Bairros
        if geojson_bairros:
            st.markdown('<div class="section-title">Distribuicao por Bairro</div>',
                        unsafe_allow_html=True)
            bairros = [ponto_no_poligono(p["lat"], p["lon"], geojson_bairros)
                       for p in st.session_state.pontos_coleta]
            contagem = pd.DataFrame({"Bairro": bairros})["Bairro"].value_counts().reset_index()
            contagem.columns = ["Bairro", "Pontos"]
            st.dataframe(contagem, use_container_width=True, hide_index=True)
            st.bar_chart(contagem.set_index("Bairro"))
    else:
        st.info("Importe pontos para gerar o relatorio.")


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer-pro">
    Logistica Profissional Goiania/GO &bull;
    OR-Tools + 2-opt + Or-opt &bull;
    Folium Maps &bull; Streamlit
</div>
""", unsafe_allow_html=True)
