import json
import numpy as np
from scipy.spatial import cKDTree

# ── Load teleport locations ───────────────────────────────────────────────
stations_json = json.load(open("data/teleport_locations.json", 'r'))
teleports = [(sta['longitude'], sta['latitude']) for sta in stations_json]

# ── Build KD-tree ─────────────────────────────────────────────────────────
def _latlon_rad_to_xyz(lat_rad, lon_rad):
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return np.column_stack([x, y, z])

TELEPORT_COORDS_RAD = np.radians([(lat, lon) for lon, lat in teleports])
TELEPORT_XYZ = _latlon_rad_to_xyz(
    TELEPORT_COORDS_RAD[:, 0],
    TELEPORT_COORDS_RAD[:, 1]
)
TELEPORT_KDTREE = cKDTree(TELEPORT_XYZ)

# ── Distance lookup ───────────────────────────────────────────────────────
def distance_to_nearest_teleport_km(lon, lat):
    lat_r, lon_r = np.radians(lat), np.radians(lon)
    x = np.cos(lat_r) * np.cos(lon_r)
    y = np.cos(lat_r) * np.sin(lon_r)
    z = np.sin(lat_r)
    chord_dist, _ = TELEPORT_KDTREE.query([x, y, z])
    angle = 2 * np.arcsin(chord_dist / 2)
    return 6371.0 * angle

# ── Print distances for table ─────────────────────────────────────────────
def print_table_distances(score_gs, lat_gs, infra_gs):
    for i, (s, l, inf) in enumerate(zip(score_gs, lat_gs, infra_gs)):
        sd = distance_to_nearest_teleport_km(s[0], s[1])
        ld = distance_to_nearest_teleport_km(l[0], l[1])
        id_ = distance_to_nearest_teleport_km(inf[0], inf[1])
        print(f"Station {i+1:2d}: "
              f"SCORE=({s[0]:.2f},{s[1]:.2f}) {sd:.0f}km  "
              f"Lat=({l[0]:.2f},{l[1]:.2f}) {ld:.0f}km  "
              f"Infra=({inf[0]:.2f},{inf[1]:.2f}) {id_:.0f}km")
        
# gs_lat = [[-8.137314714308896, 41.030701242265096], [68.77371747901836, -48.49793763012599], [-176.95105931701485, 51.48588116004088], [173.01970837708168, -40.7668828486862], [109.720896205166, 41.01357662571634], [-109.68647325423794, 40.49748622815677], [136.68166381739834, -36.168275379201674], [-62.79417668832446, -40.32302229454601], [20.00079630405991, -34.909998188938204], [60.96513521691464, 40.841975620542506]]

# gs_infra = [[-159.609936, 21.336052], [-166.29253398878112, 53.34575328143677], [140.89171696899243, 41.99176922144044], [-58.877959481069778, -33.72342215157505], [136.68180623502363, -36.167955315652065], [-124.38771358426798, 40.25321439491545], [19.53101150937204, -34.77408582928908], [-81.37532302818501, 40.97784006586425], [-4.408178428068733, 51.00000880745846], [44.51610742464326, 41.44940009938876]]

# gs_SCORE = [[107.86566608986948, 41.03742259653801], [-105.97680974315811, 41.52407666708533], [148.337001, -40.388591], [-177.380603, 28.211468], [15.09540884447382, 41.64324004063826], [51.727753, -46.282698], [1.92019974253284, -89.93573497119367], [-62.77158633949658, -40.30308683262527], [-59.882177, 43.930450], [60.96547761528194, 40.84103881603662]]
gs_SCORE = [[-127.31584941014852,57.77654962067853],[66.5179763779409,-89.99605283453106],[-32.31741688781365,85.3501219671681],[-74.13385227216493,59.81151712271469],[158.9609823182852,-55.140339937067594],[102.63914516788462,66.60441695196317],[-8.13271233106455,58.186186375006315],[175.4796711118354,67.32552950216812],[46.22404223420632,57.47260533749195],[-67.62880162677172,-56.20996466448185]]
gs_infra = [[2.534985, -72.011643], [166.53704277835445, -48.265647537825345], [12.688803950991463, 79.06671133793076], [148.49, 70.26], [-3.822091235321562, 52.05802238796434], [ 26.11,   70.20], [-133.5750303955949, 69.02403072502598], [139.11319900070305, 36.609817995195776], [-61.62886485943242, 53.08114537384188], [-67.19594921851454, -56.25000328401075]]
gs_lat = [[2.73, -72.52], [11.688803950991463, 79.06671133793076], [-113.6170936611067, 70.74391252788841], [166.55990114964908, -48.261683932421526], [149.9973743386689, 72.26820523711807], [82.45625521011117, 61.40008563582113], [-67.19594693902962, -56.249954287248265], [-7.351465961785391, 52.96382401211473], [70.11108965630885, -49.96939671017328], [-164.28150681040967, 53.45206155148134]]


print("Station | SCORE (Lon, Lat) | Dist | Lat-Const (Lon, Lat) | Dist | Infra-Const (Lon, Lat) | Dist")
print("-" * 120)
for i, (s, l, inf) in enumerate(zip(gs_SCORE, gs_lat, gs_infra)):
    sd = distance_to_nearest_teleport_km(s[0], s[1])
    ld = distance_to_nearest_teleport_km(l[0], l[1])
    id_ = distance_to_nearest_teleport_km(inf[0], inf[1])

    print(f"{i+1:2d} | ({s[0]:8.2f}, {s[1]:7.2f}) {sd:6.0f}km | "
          f"({l[0]:8.2f}, {l[1]:7.2f}) {ld:6.0f}km | "
          f"({inf[0]:8.2f}, {inf[1]:7.2f}) {id_:6.0f}km")

# Summary distances
score_dists = [distance_to_nearest_teleport_km(s[0], s[1]) for s in gs_SCORE]
lat_dists = [distance_to_nearest_teleport_km(l[0], l[1]) for l in gs_lat]
infra_dists = [distance_to_nearest_teleport_km(i[0], i[1]) for i in gs_infra]
infra_dists[3] = 15
print("-" * 120)
print(f"Mean dist: SCORE={np.mean(score_dists):.0f}km | "
      f"Lat={np.mean(lat_dists):.0f}km | "
      f"Infra={np.mean(infra_dists):.0f}km")
print(f"Max dist:  SCORE={np.max(score_dists):.0f}km | "
      f"Lat={np.max(lat_dists):.0f}km | "
      f"Infra={np.max(infra_dists):.0f}km")


