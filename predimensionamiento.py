# predimensionamiento.py
import math

# ---------- CONSTANTES Y CONVERSIONES ----------
# Resistencia del concreto típica (kg/cm²)
FC_HORMIGON = 250
FY_ACERO = 4200

# ---------- VIGAS DE HORMIGÓN (Manual MINDUR + COVENIN 1753) ----------
def predimensionar_viga_hormigon(L, w, f_c=FC_HORMIGON, f_y=FY_ACERO, zona_sismica=True):
    """
    Predimensiona viga de hormigón armado según Manual MINDUR 1985 y COVENIN 1753.
    
    Parámetros:
        L (m): Luz libre entre apoyos
        w (kg/m): Carga total (permanente + variable)
        f_c (kg/cm²): Resistencia del concreto
        f_y (kg/cm²): Fluencia del acero
        zona_sismica (bool): Si es zona sísmica (requiere más ductilidad)
    
    Retorna:
        dict: Sección propuesta y verificaciones
    """
    # 1. Momento último (simplemente apoyada)
    Mu = (w * L**2) / 8  # kg·m

    # 2. Estimación de peralte económico (fórmula ACI adaptada)
    phi = 0.9
    b = 25  # ancho inicial en cm (mínimo constructivo)
    omega = 0.18  # cuantía económica

    Mu_kgcm = Mu * 100  # convertir a kg·cm
    d = (Mu_kgcm / (phi * b * f_c * omega * (1 - 0.59 * omega))) ** 0.5
    recubrimiento = 5  # cm (para 2 horas de fuego)
    h = d + recubrimiento

    # 3. Relación ancho/alto (debe estar entre 0.3 y 0.5)
    relacion_b_h = b / h
    if relacion_b_h < 0.3:
        b = h * 0.35
    elif relacion_b_h > 0.5:
        b = h * 0.45

    # 4. Esbeltez (L/h) – rango recomendado 10-15
    L_cm = L * 100
    esbeltez = L_cm / h
    if esbeltez > 15:
        return {"error": f"Esbeltez {esbeltez:.1f} > 15 – AUMENTAR PERALTE"}
    elif esbeltez < 10:
        nota_esbeltez = f"Esbeltez baja ({esbeltez:.1f}) – podría reducirse peralte"
    else:
        nota_esbeltez = f"Esbeltez aceptable ({esbeltez:.1f})"

    # 5. Flecha (simplificada) – Manual MINDUR Cap.6
    E = 15000 * (f_c ** 0.5)  # módulo de elasticidad (kg/cm²)
    I = (b * h**3) / 12  # momento de inercia (cm⁴)
    w_kgcm = w / 100  # convertir a kg/cm
    delta = (5 * w_kgcm * L_cm**4) / (384 * E * I)  # cm
    delta_adm = L_cm / 250  # según COVENIN 2002
    if delta > delta_adm:
        return {"error": f"Flecha {delta:.2f} cm > admisible {delta_adm:.2f} cm – AUMENTAR SECCIÓN"}

    # 6. Corte (simplificado)
    Vu = w * L / 2  # corte máximo en apoyos (kg)
    Vc = 0.53 * (f_c ** 0.5) * b * d  # resistencia del concreto (kg)
    necesita_estribos = Vu > Vc

    # 7. Zona sísmica (COVENIN 1756)
    if zona_sismica:
        if relacion_b_h < 0.3:
            b = h * 0.35  # ajuste final
        # Otras verificaciones de ductilidad se harían en diseño detallado

    return {
        "seccion_propuesta": f"{b:.0f} cm × {h:.0f} cm",
        "peralte_efectivo": f"{d:.1f} cm",
        "momento_ultimo": f"{Mu:.2f} kg·m",
        "esbeltez": f"{esbeltez:.1f}",
        "nota_esbeltez": nota_esbeltez,
        "flecha": f"{delta:.2f} cm",
        "flecha_admisible": f"{delta_adm:.2f} cm",
        "corte_requiere_estribos": necesita_estribos,
        "normativa_aplicada": "Manual MINDUR 1985, COVENIN 1753, COVENIN 2002",
        "observaciones": "Predimensionado preliminar – requiere cálculo detallado"
    }

# ---------- COLUMNAS DE HORMIGÓN (COVENIN 1753 + Manual MINDUR) ----------
def predimensionar_columna_hormigon(P, L_col, f_c=FC_HORMIGON, zona_sismica=True):
    """
    Predimensiona columna de hormigón armado.

    Parámetros:
        P (kg): Carga total aproximada sobre la columna (puede estimarse como área de influencia × pisos × 1000 kg/m²)
        L_col (m): Altura libre de la columna
        f_c (kg/cm²): Resistencia del concreto
        zona_sismica (bool): Aplica criterios de ductilidad

    Retorna:
        dict: Sección propuesta y verificaciones
    """
    # 1. Área de concreto necesaria (aproximación por compresión simple)
    # Se estima que el concreto resiste ~ 0.3 f_c (incluyendo acero mínimo)
    area_necesaria = P / (0.3 * f_c)  # cm²

    # 2. Sección cuadrada mínima
    lado = math.sqrt(area_necesaria)
    # Redondear a múltiplos de 5 cm
    lado = math.ceil(lado / 5) * 5
    b = lado
    h = lado

    # 3. Verificar esbeltez (kl/r) – para pórticos arriostrados, k ≈ 1
    r = 0.3 * lado  # radio de giro aproximado para sección rectangular
    esbeltez = L_col * 100 / r
    if esbeltez > 100:
        return {"error": f"Esbeltez {esbeltez:.1f} > 100 – AUMENTAR SECCIÓN o arriostrar"}

    # 4. Zona sísmica: lado mínimo 30 cm y relación b/h ≤ 3
    if zona_sismica:
        if lado < 30:
            lado = 30
        if b > 3 * h or h > 3 * b:
            # Ajustar a cuadrada
            lado = max(b, h)
            b = lado
            h = lado

    return {
        "seccion_propuesta": f"{b:.0f} cm × {h:.0f} cm",
        "area_concreto": f"{b*h:.0f} cm²",
        "esbeltez": f"{esbeltez:.1f}",
        "carga_estimada": f"{P:.0f} kg",
        "normativa": "COVENIN 1753, Manual MINDUR",
        "observaciones": "Predimensionado – requiere cálculo de acero y pandeo"
    }

# ---------- LOSAS DE HORMIGÓN (Manual MINDUR Cap.7) ----------
def predimensionar_losa(L_menor, tipo="maciza", sobrecarga=300):
    """
    Predimensiona losa de hormigón armado.

    Parámetros:
        L_menor (m): Luz menor de la losa
        tipo (str): 'maciza', 'nervada' o 'reticular'
        sobrecarga (kg/m²): Carga variable de uso

    Retorna:
        dict: Espesor recomendado y verificaciones
    """
    L_cm = L_menor * 100
    if tipo == "maciza":
        # Espesor mínimo para losas macizas (L/40 a L/30)
        espesor = L_cm / 35  # promedio
        espesor = math.ceil(espesor / 5) * 5
        verificacion = f"Losa maciza: espesor mínimo por norma {L_cm/40:.0f}–{L_cm/30:.0f} cm"
    elif tipo == "nervada":
        # Losas nervadas (alivianadas) – espesor mínimo L/25
        espesor = L_cm / 25
        espesor = math.ceil(espesor / 5) * 5
        verificacion = "Losa nervada: requiere nervios cada 50–70 cm y losa superior de 5 cm"
    elif tipo == "reticular":
        # Losas reticulares (casetonadas) – espesor mínimo L/30
        espesor = L_cm / 30
        espesor = math.ceil(espesor / 5) * 5
        verificacion = "Losa reticular: requiere nervios en ambas direcciones y losa superior"
    else:
        return {"error": "Tipo de losa no reconocido"}

    # Verificar sobrecarga (COVENIN 2002)
    if sobrecarga > 500:
        # Aumentar espesor un 20%
        espesor = int(espesor * 1.2 / 5) * 5

    return {
        "espesor_recomendado": f"{espesor:.0f} cm",
        "tipo": tipo,
        "sobrecarga": f"{sobrecarga} kg/m²",
        "verificacion": verificacion,
        "normativa": "Manual MINDUR Cap.7, COVENIN 2002"
    }

# ---------- ZAPATAS AISLADAS ----------
def predimensionar_zapata(P, q_adm, f_c=FC_HORMIGON):
    """
    Predimensiona zapata aislada de hormigón armado.

    Parámetros:
        P (kg): Carga de la columna
        q_adm (kg/cm²): Capacidad admisible del suelo
        f_c (kg/cm²): Resistencia del concreto

    Retorna:
        dict: Dimensiones en planta y altura
    """
    # Área necesaria de zapata
    A = P / q_adm  # cm²
    lado = math.sqrt(A)
    lado = math.ceil(lado / 5) * 5  # redondear a múltiplo de 5 cm
    B = lado
    L = lado

    # Altura de zapata (aproximación: h ≈ lado/4, mínimo 30 cm)
    h = max(30, lado / 4)
    h = math.ceil(h / 5) * 5

    return {
        "dimensiones": f"{B:.0f} cm × {L:.0f} cm",
        "altura": f"{h:.0f} cm",
        "area": f"{B*L:.0f} cm²",
        "carga_columna": f"{P:.0f} kg",
        "capacidad_suelo": f"{q_adm:.2f} kg/cm²"
    }

# ---------- PEDESTALES ----------
def predimensionar_pedestal(P, f_c=FC_HORMIGON):
    """
    Predimensiona pedestal (elemento corto de transición).
    Por esbeltez, se verifica que altura ≤ 3× lado menor.
    """
    # Área necesaria (similar a columna)
    area = P / (0.3 * f_c)
    lado = math.sqrt(area)
    lado = math.ceil(lado / 5) * 5
    # Altura máxima permitida (3× lado)
    h_max = 3 * lado
    return {
        "seccion_sugerida": f"{lado:.0f} cm × {lado:.0f} cm",
        "altura_maxima": f"{h_max:.0f} cm",
        "observaciones": "Verificar que la altura real ≤ altura máxima"
    }

# ---------- ESCALERAS (Manual MINDUR Cap.8) ----------
def predimensionar_escalera(L_inclinada, tipo="recta"):
    """
    Predimensiona escalera de hormigón armado.

    Parámetros:
        L_inclinada (m): Longitud inclinada de la escalera
        tipo (str): 'recta', 'helicoidal'

    Retorna:
        dict: Espesor recomendado
    """
    L_cm = L_inclinada * 100
    if tipo == "recta":
        espesor = L_cm / 25
    elif tipo == "helicoidal":
        espesor = L_cm / 20  # más robusta
    else:
        return {"error": "Tipo no soportado"}
    espesor = math.ceil(espesor / 5) * 5
    return {
        "espesor_recomendado": f"{espesor:.0f} cm",
        "tipo": tipo,
        "normativa": "Manual MINDUR Cap.8"
    }

# ---------- FUNCIÓN DE ALTO NIVEL (para llamar desde herramientas) ----------
def predimensionar_estructura(tipo, **kwargs):
    """
    Función general que redirige a la función específica según el tipo.
    """
    tipo = tipo.lower()
    if tipo == "viga":
        return predimensionar_viga_hormigon(**kwargs)
    elif tipo == "columna":
        return predimensionar_columna_hormigon(**kwargs)
    elif tipo == "losa":
        return predimensionar_losa(**kwargs)
    elif tipo == "zapata":
        return predimensionar_zapata(**kwargs)
    elif tipo == "pedestal":
        return predimensionar_pedestal(**kwargs)
    elif tipo == "escalera":
        return predimensionar_escalera(**kwargs)
    else:
        return {"error": f"Tipo '{tipo}' no implementado"}
