"""
algoritmo_YOLO.py
-----------------------
Pipeline final de deteccion de ganglios YOLO 2.5D multiplano sobre el VOLUMEN
ENTERO (sin mascara GT, sin leakage). Es el modo de despliegue real.

Para cada paciente:
  1. Infiere el YOLO de cada plano (axial, coronal, sagital) sobre todos los
     cortes del volumen, con una confianza propia por plano.
  2. Lleva cada caja 2D a su VOI 3D deshaciendo el rot90+fliplr.
  3. Cruza los tres planos (interseccion TRIPLE): un VOI necesita coincidir en
     axial, coronal y sagital a la vez.
  4. Fusiona los VOIs muy solapados y suprime los contenidos en otro mayor.
  5. Guarda los VOIs (txt), el ROI para 3D Slicer (.mrk.json) y un plot 3D.

NO evalua ni calcula metricas: solo genera los ficheros de salida. La
evaluacion por folds va en su script aparte (07_metricas_YOLO).

Confianza por plano: axial mas alto (plano fuerte, no mete FP), coronal y
sagital mas bajos (planos flojos, recuperan detecciones debiles). El cruce
triple absorbe los FP que eso pueda generar.
"""

from pathlib import Path
import nibabel as nib
import numpy as np
import plotly.graph_objects as go
from ultralytics import YOLO
import json


# ---- parametros de fusion (bloqueados, no cambiar sin repetir la validacion) ----
gap = 1          # separacion de cortes para el 2.5D (canal previo y siguiente)
t = 4            # holgura del casado entre planos
t_voi = 8        # margen de supresion de VOIs contenidos en otro
iou_thr = 0.3    # umbral de fusion de VOIs solapados

# ---- confianza POR PLANO ----
conf_ax = 0.10   # axial (plano fuerte)
conf_co = 0.05   # coronal (plano flojo)
conf_sa = 0.05   # sagital (plano flojo)

planes = {0: "Sagital", 1: "Coronal", 2: "Axial"}


# ======================================================================
#  EXTRACCION DE CORTES Y NORMALIZACION
# ======================================================================

def get_slice(vol, axis, i):
    """
    Devuelve el corte i del volumen a lo largo de un eje, ya rotado y volteado
    como espera el YOLO (rot90 + fliplr). axis: 0=sagital, 1=coronal, 2=axial.
    """
    if axis == 0:
        sl = vol[i, :, :]
    elif axis == 1:
        sl = vol[:, i, :]
    else:
        sl = vol[:, :, i]

    sl = np.rot90(sl)
    sl = np.fliplr(sl)
    return sl


def get_slice_25d(vol, axis, i, gap):
    """
    Devuelve tres cortes (i-gap, i, i+gap) para montar la imagen 2.5D: el corte
    central y sus dos vecinos como canales. En los bordes se repite el extremo.
    """
    n = vol.shape[axis]
    i_prev = max(0, i - gap)
    i_next = min(n - 1, i + gap)

    s_prev = get_slice(vol, axis, i_prev)
    s_current = get_slice(vol, axis, i)
    s_next = get_slice(vol, axis, i_next)
    return s_prev, s_current, s_next


def norm_hu(img_in):
    """
    Normaliza unidades Hounsfield al rango [-150, 250] y lo lleva a 0-255 uint8.
    Esa ventana resalta tejidos blandos y ganglios.
    """
    hu_min, hu_max = -150, 250
    norm = ((img_in - hu_min) / (hu_max - hu_min) * 255).clip(0, 255).astype(np.uint8)
    return norm


# ======================================================================
#  INFERENCIA POR PLANO
# ======================================================================

def predict_plane(img_vol, axis, model, conf, gap):
    """
    Pasa el YOLO del plano sobre TODOS los cortes del volumen (full volume).
    Devuelve {indice_corte: [(x1,y1,x2,y2), ...]} en espacio imagen.
    """
    n = img_vol.shape[axis]
    boxes_por_corte = {}

    for i in range(n):
        s_prev, s_curr, s_next = get_slice_25d(img_vol, axis, i, gap)
        rgb = np.stack([norm_hu(s_prev), norm_hu(s_curr), norm_hu(s_next)], axis=-1)

        results = model(rgb, conf=conf, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()

        if len(boxes) > 0:
            boxes_por_corte[i] = [tuple(map(float, b)) for b in boxes]

    return boxes_por_corte


# ======================================================================
#  PASO DE CAJA 2D A VOI 3D
# ======================================================================

def inverse_box(box_img, H, W):
    """
    Deshace el rot90 y el fliplr aplicados al generar la imagen. Devuelve la
    caja en indices del array original como (fmin, cmin, fmax, cmax).
    """
    x1, y1, x2, y2 = box_img

    f_a = (H - 1) - x1
    c_a = (W - 1) - y1
    f_b = (H - 1) - x2
    c_b = (W - 1) - y2

    fmin, fmax = min(f_a, f_b), max(f_a, f_b)
    cmin, cmax = min(c_a, c_b), max(c_a, c_b)
    return fmin, cmin, fmax, cmax


def box_2d_to_voi(box_inv, i, axis):
    """
    Coloca una caja 2D (ya invertida) y su indice de corte en el espacio 3D.
    El indice del corte ocupa el eje que ese plano recorre, quedando fino ahi.
    Devuelve (x_min, x_max, y_min, y_max, z_min, z_max).
    """
    fmin, cmin, fmax, cmax = box_inv

    if axis == 2:      # axial: fino en z
        x_min, x_max = fmin, fmax
        y_min, y_max = cmin, cmax
        z_min, z_max = i, i
    elif axis == 1:    # coronal: fino en y
        x_min, x_max = fmin, fmax
        y_min, y_max = i, i
        z_min, z_max = cmin, cmax
    else:              # sagital: fino en x
        x_min, x_max = i, i
        y_min, y_max = fmin, fmax
        z_min, z_max = cmin, cmax

    return x_min, x_max, y_min, y_max, z_min, z_max


def boxes_to_vois(boxes_por_corte, axis, H, W):
    """
    Convierte la salida de predict_plane en una lista de VOIs 3D de un plano.
    Para cada caja de cada corte deshace la transformacion y la coloca en 3D.
    """
    lista_vois = []
    for i, boxes in boxes_por_corte.items():
        for box_img in boxes:
            box_inv = inverse_box(box_img, H, W)
            voi = box_2d_to_voi(box_inv, i, axis)
            lista_vois.append(voi)
    return lista_vois


# ======================================================================
#  CRUCE TRIPLE DE LOS TRES PLANOS
# ======================================================================

def assemble_voi(ax, co, sa):
    """
    Construye la caja final tomando de cada eje el rango de los dos planos que
    ven ese eje (min de ambos como inicio, max como fin).
    x lo ven axial+coronal | y axial+sagital | z coronal+sagital.
    """
    axn, axx, ayn, ayx, azn, azx = ax
    cxn, cxx, cyn, cyx, czn, czx = co
    sxn, sxx, syn, syx, szn, szx = sa

    x_min, x_max = min(axn, cxn), max(axx, cxx)
    y_min, y_max = min(ayn, syn), max(ayx, syx)
    z_min, z_max = min(czn, szn), max(czx, szx)
    return x_min, x_max, y_min, y_max, z_min, z_max


def punto_en_rango(p, lo, hi, t):
    """True si p cae dentro de [lo, hi] con tolerancia t a cada lado."""
    return (lo - t) <= p <= (hi + t)


def casan(ax, co, sa, t):
    """
    Comprueba que las tres laminas se cruzan en una region comun. Cada plano es
    fino en su eje de corte; ese valor fino tiene que caer dentro del rango de
    los otros dos planos que si ven ese eje.
    """
    axn, axx, ayn, ayx, azn, azx = ax
    cxn, cxx, cyn, cyx, czn, czx = co
    sxn, sxx, syn, syx, szn, szx = sa

    z_axial_en_coronal = punto_en_rango(azn, czn, czx, t)
    z_axial_en_sagital = punto_en_rango(azn, szn, szx, t)
    y_coronal_en_axial = punto_en_rango(cyn, ayn, ayx, t)
    y_coronal_en_sagital = punto_en_rango(cyn, syn, syx, t)
    x_sagital_en_axial = punto_en_rango(sxn, axn, axx, t)
    x_sagital_en_coronal = punto_en_rango(sxn, cxn, cxx, t)

    return (z_axial_en_coronal and z_axial_en_sagital and
            y_coronal_en_axial and y_coronal_en_sagital and
            x_sagital_en_axial and x_sagital_en_coronal)


def find_vois(vois_axial, vois_coronal, vois_sagital, t):
    """
    Recorre todas las combinaciones de cajas axial-coronal-sagital. Cuando un
    trio se cruza, ensambla su VOI y lo guarda evitando duplicados exactos.
    """
    lista_vois = []
    for ax in vois_axial:
        for co in vois_coronal:
            for sa in vois_sagital:
                if casan(ax, co, sa, t):
                    voi = assemble_voi(ax, co, sa)
                    if voi not in lista_vois:
                        lista_vois.append(voi)
    return lista_vois


# ======================================================================
#  GEOMETRIA: VOLUMEN, IOU, ENGLOBE
# ======================================================================

def volumen_voi(voi):
    """Volumen de un VOI como producto de sus tres lados."""
    x0, x1, y0, y1, z0, z1 = voi
    return (x1 - x0) * (y1 - y0) * (z1 - z0)


def iou_3d(voi_a, voi_b):
    """IoU 3D entre dos VOIs: interseccion / union. 0 si no se solapan."""
    ax0, ax1, ay0, ay1, az0, az1 = voi_a
    bx0, bx1, by0, by1, bz0, bz1 = voi_b

    inter_x = max(0, min(ax1, bx1) - max(ax0, bx0))
    inter_y = max(0, min(ay1, by1) - max(ay0, by0))
    inter_z = max(0, min(az1, bz1) - max(az0, bz0))
    interseccion = inter_x * inter_y * inter_z

    vol_a = (ax1 - ax0) * (ay1 - ay0) * (az1 - az0)
    vol_b = (bx1 - bx0) * (by1 - by0) * (bz1 - bz0)
    union = vol_a + vol_b - interseccion

    if union <= 0:
        return 0.0
    return interseccion / union


def voi_engloba(voi_grande, voi_pequeno, t_voi):
    """
    True si voi_grande contiene a voi_pequeno, dejando que el pequeno sobresalga
    hasta t_voi px por cada lado.
    """
    gx0, gx1, gy0, gy1, gz0, gz1 = voi_grande
    px0, px1, py0, py1, pz0, pz1 = voi_pequeno

    dentro_x = (gx0 - t_voi) <= px0 and px1 <= (gx1 + t_voi)
    dentro_y = (gy0 - t_voi) <= py0 and py1 <= (gy1 + t_voi)
    dentro_z = (gz0 - t_voi) <= pz0 and pz1 <= (gz1 + t_voi)
    return dentro_x and dentro_y and dentro_z


def union_grupo(grupo):
    """Caja minima que engloba a todos los VOIs de un grupo."""
    x_min = min(v[0] for v in grupo)
    x_max = max(v[1] for v in grupo)
    y_min = min(v[2] for v in grupo)
    y_max = max(v[3] for v in grupo)
    z_min = min(v[4] for v in grupo)
    z_max = max(v[5] for v in grupo)
    return x_min, x_max, y_min, y_max, z_min, z_max


# ======================================================================
#  FUSION Y SUPRESION DE VOIs
# ======================================================================

def fusionar_por_iou(vois, iou_thr):
    """
    Une en una sola caja los VOIs cuyo IoU supere el umbral. Repite hasta que
    ningun grupo crece (al unir dos cajas la nueva puede solapar con una tercera).
    """
    grupos = list(vois)
    cambio = True

    while cambio:
        cambio = False
        nuevos = []
        usados = [False] * len(grupos)

        for i in range(len(grupos)):
            if usados[i]:
                continue
            grupo_actual = [grupos[i]]
            usados[i] = True

            for j in range(i + 1, len(grupos)):
                if usados[j]:
                    continue
                if iou_3d(grupo_actual[0], grupos[j]) > iou_thr:
                    grupo_actual.append(grupos[j])
                    usados[j] = True
                    cambio = True

            nuevos.append(union_grupo(grupo_actual))

        grupos = nuevos

    return grupos


def suprimir_vois(vois, t_voi):
    """
    Elimina los VOIs englobados por otro mayor (con margen t_voi). No fusiona ni
    crea cajas nuevas: se queda con el que engloba, tal cual.
    """
    quedan = []
    for i in range(len(vois)):
        englobado = False
        for j in range(len(vois)):
            if i == j:
                continue
            if volumen_voi(vois[j]) < volumen_voi(vois[i]):
                continue
            if voi_engloba(vois[j], vois[i], t_voi):
                englobado = True
                break
        if not englobado:
            quedan.append(vois[i])
    return quedan


# ======================================================================
#  PIPELINE COMPLETO DE UN PACIENTE
# ======================================================================

def fusionar_paciente(img_vol, model_ax, model_co, model_sa,
                      t, conf_ax, conf_co, conf_sa, gap, t_voi, iou_thr):
    """
    Pipeline completo sobre el volumen ENTERO. Cada plano infiere con su
    confianza propia, se llevan las cajas a 3D, se cruzan los tres (triple),
    se fusionan los solapados y se suprimen los contenidos.
    Devuelve (vois_fusion, (vois_ax, vois_co, vois_sa)).
    """
    N0, N1, N2 = img_vol.shape

    boxes_ax = predict_plane(img_vol, 2, model_ax, conf_ax, gap)
    boxes_co = predict_plane(img_vol, 1, model_co, conf_co, gap)
    boxes_sa = predict_plane(img_vol, 0, model_sa, conf_sa, gap)

    vois_ax = boxes_to_vois(boxes_ax, 2, N0, N1)
    vois_co = boxes_to_vois(boxes_co, 1, N0, N2)
    vois_sa = boxes_to_vois(boxes_sa, 0, N1, N2)

    vois_fusion = find_vois(vois_ax, vois_co, vois_sa, t)
    vois_fusion = fusionar_por_iou(vois_fusion, iou_thr)   # une cruces fuertes
    vois_fusion = suprimir_vois(vois_fusion, t_voi)         # elimina contenidos

    return vois_fusion, (vois_ax, vois_co, vois_sa)


# ======================================================================
#  EXPORTACION: TXT, SLICER, PLOT
# ======================================================================

def voi_to_corners(voi):
    """Convierte un VOI en los ocho vertices de su cubo."""
    x0, x1, y0, y1, z0, z1 = voi
    return np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ])


def voi_a_roi_ras(voi, affine):
    """Pasa un VOI de indices de array a centro y tamano en RAS (mm) via affine."""
    corners_idx = voi_to_corners(voi)
    corners_h = np.hstack([corners_idx, np.ones((8, 1))])
    corners_ras = (affine @ corners_h.T).T[:, :3]

    ras_min = corners_ras.min(axis=0)
    ras_max = corners_ras.max(axis=0)
    centro = (ras_min + ras_max) / 2
    size = ras_max - ras_min
    return centro, size


def guardar_vois(vois, output_dir, paciente):
    """
    Guarda los VOIs en un txt, una linea por VOI con los seis valores separados
    por espacios (mismo formato que el ground truth).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    voi_path = output_dir / f"{paciente}_VOIs_predict.txt"

    with open(voi_path, "w") as f:
        for voi in vois:
            x_min, x_max, y_min, y_max, z_min, z_max = voi
            f.write(f"{x_min} {x_max} {y_min} {y_max} {z_min} {z_max}\n")

    print(f"Guardado: {voi_path} ({len(vois)} VOIs)")


def guardar_mrk_json(vois, affine, output_dir, paciente):
    """Genera el .mrk.json de ROIs para abrir en 3D Slicer."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{paciente}_node_predict.mrk.json"

    markups = []
    for n, voi in enumerate(vois):
        centro, size = voi_a_roi_ras(voi, affine)
        markups.append({
            "type": "ROI",
            "coordinateSystem": "RAS",
            "coordinateUnits": "mm",
            "locked": True,
            "fixedNumberOfControlPoints": False,
            "labelFormat": "%N-%d",
            "lastUsedControlPointNumber": 1,
            "roiType": "Box",
            "center": [float(centro[0]), float(centro[1]), float(centro[2])],
            "orientation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "size": [float(size[0]), float(size[1]), float(size[2])],
            "insideOut": False,
            "controlPoints": [{
                "id": str(n + 1),
                "label": f"node_{n + 1}",
                "description": "",
                "associatedNodeID": "",
                "position": [float(centro[0]), float(centro[1]), float(centro[2])],
                "orientation": [-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0],
                "selected": True,
                "locked": False,
                "visibility": True,
                "positionStatus": "defined",
            }],
            "measurements": [{
                "name": "volume",
                "enabled": False,
                "units": "cm3",
                "printFormat": "%-#4.4g%s",
            }],
            "display": {
                "visibility": True,
                "opacity": 0.8,
                "color": [1.0, 1.0, 0.0],
                "selectedColor": [1.0, 0.5000076295109483, 0.5000076295109483],
                "activeColor": [0.4, 1.0, 0.0],
                "propertiesLabelVisibility": False,
                "pointLabelsVisibility": False,
                "textScale": 0.0,
                "glyphType": "Sphere3D",
                "glyphScale": 0.0,
                "glyphSize": 1.0,
                "useGlyphScale": True,
                "sliceProjection": False,
                "sliceProjectionUseFiducialColor": True,
                "sliceProjectionOutlinedBehindSlicePlane": False,
                "sliceProjectionColor": [1.0, 1.0, 1.0],
                "sliceProjectionOpacity": 0.6,
                "lineThickness": 0.2200000000000004,
                "lineColorFadingStart": 1.0,
                "lineColorFadingEnd": 10.0,
                "lineColorFadingSaturation": 1.0,
                "lineColorFadingHueOffset": 0.0,
                "handlesInteractive": False,
                "translationHandleVisibility": False,
                "rotationHandleVisibility": False,
                "scaleHandleVisibility": False,
                "interactionHandleScale": 3.0,
                "snapMode": "toVisibleSurface",
            },
        })

    contenido = {
        "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/"
                   "Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
        "markups": markups,
    }

    with open(json_path, "w") as f:
        json.dump(contenido, f, indent=2)

    print(f"Guardado: {json_path} ({len(markups)} ROIs)")


def cargar_gt_voi(gt_root, paciente):
    """
    Lee el GT VOI de un paciente (solo para dibujarlo en el plot, opcional).
    Devuelve la tupla o None si no existe.
    """
    gt_path = Path(gt_root) / paciente / f"gt_voi_{paciente}.txt"
    if not gt_path.exists():
        return None
    with open(gt_path, "r") as f:
        valores = f.read().split()
    return tuple(int(float(v)) for v in valores[:6])


def plot_vois_3d(vois_por_plano, vois_fusion, paciente, out_html, gt_voi=None):
    """
    Dibuja las laminas de cada plano como caras semitransparentes (axial rojo,
    coronal verde, sagital azul) y los VOIs fusionados como aristas amarillas.
    Si se pasa gt_voi, lo dibuja en naranja.
    """
    vois_ax, vois_co, vois_sa = vois_por_plano

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    def cara_opaca(voi, color):
        x0, x1, y0, y1, z0, z1 = voi
        if x0 == x1:        # plano sagital
            xs = [x0, x0, x0, x0]; ys = [y0, y1, y1, y0]; zs = [z0, z0, z1, z1]
        elif y0 == y1:      # plano coronal
            xs = [x0, x1, x1, x0]; ys = [y0, y0, y0, y0]; zs = [z0, z0, z1, z1]
        else:               # plano axial
            xs = [x0, x1, x1, x0]; ys = [y0, y0, y1, y1]; zs = [z0, z0, z0, z0]
        return go.Mesh3d(x=xs, y=ys, z=zs, color=color, opacity=0.5,
                         i=[0, 0], j=[1, 2], k=[2, 3], showlegend=False)

    def aristas(voi, color, width):
        c = voi_to_corners(voi)
        trazas = []
        for a, b in edges:
            trazas.append(go.Scatter3d(
                x=[c[a][0], c[b][0]], y=[c[a][1], c[b][1]], z=[c[a][2], c[b][2]],
                mode="lines", line=dict(color=color, width=width), showlegend=False))
        return trazas

    traces = []
    for voi in vois_ax:
        traces.append(cara_opaca(voi, "red"))
    for voi in vois_co:
        traces.append(cara_opaca(voi, "green"))
    for voi in vois_sa:
        traces.append(cara_opaca(voi, "blue"))
    for voi in vois_fusion:
        traces += aristas(voi, "rgb(255,255,0)", 6)
    if gt_voi is not None:
        traces += aristas(gt_voi, "rgb(255,102,0)", 7)

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(xaxis=dict(title="X"), yaxis=dict(title="Y"),
                   zaxis=dict(title="Z"), aspectmode="data"),
        title=f"Planos y VOIs - Paciente {paciente}",
        showlegend=False,
    )
    fig.write_html(out_html, auto_open=False)
    print(f"Plot guardado: {out_html}")


# ======================================================================
#  EJECUCION (demo sobre un conjunto de pacientes, un solo modelo por plano)
# ======================================================================

if __name__ == "__main__":

    plot_flag_3D = True
    plot_gt = True   # dibujar el GT en el plot si existe (solo visual)

    pacientes_root = Path(r"C:\Users\diego\Desktop\TFM\Pacientes_Preprocesados_TFM")
    output_vois = Path(r"C:\Users\diego\Desktop\TFM\VOIs_predict")
    gt_root = Path(r"C:\Users\diego\Desktop\TFM\GT_VOIs")
    plot_dir = Path(r"C:\Users\diego\Desktop\TFM\VOIs_predict\plots")

    model_ax = YOLO(r"C:\Users\diego\Desktop\TFM\FineTuning_YOLO11s_KFold_results"
                    r"\YOLO11s_finetune_Axial_fold0\weights\best.pt")
    model_co = YOLO(r"C:\Users\diego\Desktop\TFM\FineTuning_YOLO11s_KFold_results"
                    r"\YOLO11s_finetune_Coronal_fold0\weights\best.pt")
    model_sa = YOLO(r"C:\Users\diego\Desktop\TFM\FineTuning_YOLO11s_KFold_results"
                    r"\YOLO11s_finetune_Sagital_fold0\weights\best.pt")

    test_pacientes = ['054', '049', '051', '164', '078', '092', '010', '030',
                      '178', '068', '109', '166', '019', '163', '035', '123', '125']

    for paciente in test_pacientes:
        print(f"Procesando {paciente}")

        img_path = pacientes_root / paciente / "image_resampled_process.nii.gz"
        nii = nib.load(str(img_path))
        img_vol = nii.get_fdata()
        affine = nii.affine

        vois_fusion, vois_por_plano = fusionar_paciente(
            img_vol, model_ax, model_co, model_sa,
            t, conf_ax, conf_co, conf_sa, gap, t_voi, iou_thr)

        guardar_vois(vois_fusion, output_vois, paciente)
        guardar_mrk_json(vois_fusion, affine, output_vois, paciente)

        if plot_flag_3D:
            plot_dir.mkdir(parents=True, exist_ok=True)
            out_html = plot_dir / f"{paciente}_vois.html"
            gt_voi = cargar_gt_voi(gt_root, paciente) if plot_gt else None
            plot_vois_3d(vois_por_plano, vois_fusion, paciente, str(out_html), gt_voi)
