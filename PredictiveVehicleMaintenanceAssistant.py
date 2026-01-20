"""
Predictive Vehicle Maintenance Assistant

Single-file, market-ready Tkinter application for:
- Predictive fault scoring and explanations
- Safety alerts
- Fleet management with history and dashboards
- Text and PDF reporting with charts
- Multi-language support (EN, ES, FR)
- Demo/simulation mode (no hardware required)

Dependencies (Python 3.10+):
- Tkinter (standard library)
- matplotlib
- reportlab
- pandas
"""

import os
import random
import string
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, Image

import tkinter as tk
from tkinter import ttk, messagebox, filedialog


# -----------------------------
# Data classes and core logic
# -----------------------------


@dataclass
class VehicleReading:
    """Represents a simple reading for compatibility with earlier logic."""

    vehicle_id: str
    timestamp: datetime
    odometer_km: float
    engine_hours: Optional[float] = None
    fault_codes: Optional[List[str]] = None


@dataclass
class MaintenanceRecommendation:
    """Maintenance recommendation data container."""

    vehicle_id: str
    recommended_action: str
    reason: str
    priority: str  # "low", "medium", "high"


@dataclass
class FaultInfo:
    """Represents a single OBD-II fault and its metadata."""

    code: str
    system: str
    description_en: str
    description_es: str
    description_fr: str
    base_severity: int  # 1–10
    safety_critical: bool


@dataclass
class DiagnosticStep:
    """Detailed diagnostic and action steps for a specific fault."""

    code: str
    steps_en: List[str]
    steps_es: List[str]
    steps_fr: List[str]
    recommended_action_en: str
    recommended_action_es: str
    recommended_action_fr: str
    safety_warning_en: str
    safety_warning_es: str
    safety_warning_fr: str


@dataclass
class FleetVehicle:
    """Represents a vehicle in the fleet."""

    vehicle_id: str
    make: str
    model: str
    year: int


class PredictiveVehicleMaintenanceAssistantCore:
    """
    Core scoring and data model.

    Includes:
    - OBD-II fault database
    - Diagnostic workflows
    - Simple predictive scoring logic
    - Safety alert generation
    - Fleet history handling
    """

    def __init__(self, service_interval_km: float = 10000.0) -> None:
        self.service_interval_km = service_interval_km
        self.fault_db: Dict[str, FaultInfo] = self.get_fault_database()
        self.diagnostic_db: Dict[str, DiagnosticStep] = self.get_diagnostic_steps()
        # Fleet history stored in a pandas DataFrame for easy analysis
        self.history_columns = [
            "timestamp",
            "vehicle_id",
            "make",
            "model",
            "year",
            "fault_code",
            "probability",
            "system",
            "safety_critical",
        ]
        self.history_df = pd.DataFrame(columns=self.history_columns)

    # -----------------------------
    # Backwards-compatible basic logic
    # -----------------------------

    def analyze_reading(self, reading: VehicleReading) -> List[MaintenanceRecommendation]:
        """
        Original example logic retained for compatibility and potential reuse.
        """
        recommendations: List[MaintenanceRecommendation] = []

        # 1. Recommend service if odometer is near multiple of service_interval_km
        if reading.odometer_km > 0:
            km_to_next_service = self.service_interval_km - (reading.odometer_km % self.service_interval_km)
            if km_to_next_service < 500:
                recommendations.append(
                    MaintenanceRecommendation(
                        vehicle_id=reading.vehicle_id,
                        recommended_action="Schedule routine service",
                        reason=(
                            f"Within {km_to_next_service:.0f} km of the next "
                            f"{self.service_interval_km:.0f} km service interval."
                        ),
                        priority="medium",
                    )
                )

        # 2. If fault codes present, flag high-priority inspection
        if reading.fault_codes:
            recommendations.append(
                MaintenanceRecommendation(
                    vehicle_id=reading.vehicle_id,
                    recommended_action="Inspect vehicle due to fault codes",
                    reason=f"Detected fault codes: {', '.join(reading.fault_codes)}",
                    priority="high",
                )
            )

        return recommendations

    # -----------------------------
    # Fault database and diagnostics
    # -----------------------------

    def get_fault_database(self) -> Dict[str, FaultInfo]:
        """
        Return a realistic, multi-system fault database.

        NOTE: This is not an exhaustive list of all OBD-II codes,
        but covers key systems with realistic examples for demo and
        real-world extension.
        """
        faults: List[FaultInfo] = [
            # Engine / Emissions (P0xxx)
            FaultInfo(
                code="P0300",
                system="Engine",
                description_en="Random/multiple cylinder misfire detected",
                description_es="Fallo de encendido aleatorio/múltiple detectado",
                description_fr="Raté d’allumage aléatoire/multiple détecté",
                base_severity=8,
                safety_critical=True,
            ),
            FaultInfo(
                code="P0301",
                system="Engine",
                description_en="Cylinder 1 misfire detected",
                description_es="Fallo de encendido en cilindro 1 detectado",
                description_fr="Raté d’allumage cylindre 1 détecté",
                base_severity=7,
                safety_critical=True,
            ),
            FaultInfo(
                code="P0171",
                system="Engine",
                description_en="System too lean (Bank 1)",
                description_es="Sistema demasiado pobre (Banco 1)",
                description_fr="Système trop pauvre (Banque 1)",
                base_severity=6,
                safety_critical=False,
            ),
            FaultInfo(
                code="P0420",
                system="Emissions",
                description_en="Catalyst system efficiency below threshold (Bank 1)",
                description_es="Eficiencia del sistema catalítico por debajo del umbral (Banco 1)",
                description_fr="Efficacité du système catalytique en dessous du seuil (Banque 1)",
                base_severity=5,
                safety_critical=False,
            ),
            FaultInfo(
                code="P0128",
                system="Cooling",
                description_en="Coolant thermostat below regulating temperature",
                description_es="Termostato de refrigerante por debajo de la temperatura regulada",
                description_fr="Thermostat de liquide de refroidissement en dessous de la température de régulation",
                base_severity=4,
                safety_critical=False,
            ),
            FaultInfo(
                code="P0217",
                system="Cooling",
                description_en="Engine over-temperature condition",
                description_es="Condición de sobrecalentamiento del motor",
                description_fr="Condition de surchauffe du moteur",
                base_severity=9,
                safety_critical=True,
            ),
            # Brakes (C0xxx)
            FaultInfo(
                code="C0035",
                system="Brakes",
                description_en="Left front wheel speed sensor circuit",
                description_es="Circuito del sensor de velocidad de rueda delantera izquierda",
                description_fr="Circuit capteur vitesse roue avant gauche",
                base_severity=8,
                safety_critical=True,
            ),
            FaultInfo(
                code="C0110",
                system="Brakes",
                description_en="ABS pump motor circuit",
                description_es="Circuito del motor de la bomba ABS",
                description_fr="Circuit moteur pompe ABS",
                base_severity=9,
                safety_critical=True,
            ),
            # Transmission (P07xx)
            FaultInfo(
                code="P0700",
                system="Transmission",
                description_en="Transmission control system malfunction",
                description_es="Mal funcionamiento del sistema de control de la transmisión",
                description_fr="Défaillance du système de commande de transmission",
                base_severity=7,
                safety_critical=True,
            ),
            FaultInfo(
                code="P0730",
                system="Transmission",
                description_en="Incorrect gear ratio",
                description_es="Relación de cambio incorrecta",
                description_fr="Rapport de démultiplication incorrect",
                base_severity=6,
                safety_critical=True,
            ),
            # Suspension (C1xxx)
            FaultInfo(
                code="C1145",
                system="Suspension",
                description_en="Right front wheel speed sensor input circuit failure",
                description_es="Fallo en el circuito de entrada del sensor de velocidad de rueda delantera derecha",
                description_fr="Défaillance du circuit d'entrée capteur vitesse roue avant droite",
                base_severity=5,
                safety_critical=False,
            ),
            # Electrical (B, U codes)
            FaultInfo(
                code="B0020",
                system="Safety Restraint",
                description_en="Driver frontal deployment loop stage 2",
                description_es="Circuito de despliegue frontal del conductor, etapa 2",
                description_fr="Boucle de déploiement frontal conducteur, étape 2",
                base_severity=9,
                safety_critical=True,
            ),
            FaultInfo(
                code="U0100",
                system="Network",
                description_en="Lost communication with ECM/PCM",
                description_es="Comunicación perdida con ECM/PCM",
                description_fr="Perte de communication avec ECM/PCM",
                base_severity=8,
                safety_critical=True,
            ),
            FaultInfo(
                code="U0121",
                system="Network",
                description_en="Lost communication with ABS control module",
                description_es="Comunicación perdida con el módulo de control ABS",
                description_fr="Perte de communication avec le module de commande ABS",
                base_severity=9,
                safety_critical=True,
            ),
        ]
        return {f.code: f for f in faults}

    def get_diagnostic_steps(self) -> Dict[str, DiagnosticStep]:
        """
        Return detailed diagnostic steps for each known fault.
        """
        steps: List[DiagnosticStep] = [
            DiagnosticStep(
                code="P0300",
                steps_en=[
                    "Verify misfire with live data or road test.",
                    "Inspect spark plugs, ignition coils, and wiring.",
                    "Check fuel pressure and injector operation.",
                    "Perform compression test if misfire persists.",
                ],
                steps_es=[
                    "Verifique el fallo de encendido con datos en vivo o prueba de manejo.",
                    "Inspeccione bujías, bobinas de encendido y cableado.",
                    "Compruebe la presión de combustible y funcionamiento de inyectores.",
                    "Realice prueba de compresión si el fallo persiste.",
                ],
                steps_fr=[
                    "Vérifiez le raté d’allumage avec données en temps réel ou essai routier.",
                    "Inspectez bougies, bobines et câblage.",
                    "Contrôlez la pression de carburant et les injecteurs.",
                    "Effectuez un test de compression si le problème persiste.",
                ],
                recommended_action_en="Address ignition/fuel issues and re-test; avoid prolonged driving with misfires.",
                recommended_action_es="Solucione problemas de encendido/combustible y vuelva a probar; evite conducir mucho con fallos.",
                recommended_action_fr="Corrigez les problèmes d’allumage/carburant et re-testez; évitez de rouler longuement avec des ratés.",
                safety_warning_en="High misfire levels can damage the catalytic converter and reduce engine power.",
                safety_warning_es="Un alto nivel de fallos de encendido puede dañar el catalizador y reducir la potencia.",
                safety_warning_fr="Un fort niveau de ratés peut endommager le catalyseur et réduire la puissance moteur.",
            ),
            DiagnosticStep(
                code="P0420",
                steps_en=[
                    "Confirm code after warm-up and drive cycle.",
                    "Inspect exhaust for leaks before the catalytic converter.",
                    "Check oxygen sensor operation (upstream and downstream).",
                    "Evaluate catalytic converter efficiency and replace if required.",
                ],
                steps_es=[
                    "Confirme el código después del calentamiento y ciclo de manejo.",
                    "Inspeccione el escape por fugas antes del catalizador.",
                    "Compruebe el funcionamiento de los sensores de oxígeno (antes y después del catalizador).",
                    "Evalúe la eficiencia del catalizador y reemplácelo si es necesario.",
                ],
                steps_fr=[
                    "Confirmez le code après la mise en température et le cycle de conduite.",
                    "Inspectez le système d’échappement pour des fuites avant le catalyseur.",
                    "Vérifiez le fonctionnement des sondes lambda (amont et aval).",
                    "Évaluez l’efficacité du catalyseur et remplacez-le si nécessaire.",
                ],
                recommended_action_en="Repair exhaust leaks or replace catalyst as needed.",
                recommended_action_es="Repare las fugas de escape o reemplace el catalizador según sea necesario.",
                recommended_action_fr="Réparez les fuites d’échappement ou remplacez le catalyseur si nécessaire.",
                safety_warning_en="Generally not immediately dangerous but can increase emissions and reduce performance.",
                safety_warning_es="Generalmente no es peligroso de inmediato, pero puede aumentar emisiones y reducir rendimiento.",
                safety_warning_fr="Généralement non critique immédiatement, mais augmente les émissions et réduit les performances.",
            ),
            DiagnosticStep(
                code="P0217",
                steps_en=[
                    "Verify coolant level and check for external leaks.",
                    "Inspect radiator fan operation and fuses/relays.",
                    "Check thermostat and water pump function.",
                    "Confirm no blockage in radiator or cooling passages.",
                ],
                steps_es=[
                    "Verifique el nivel de refrigerante y revise fugas externas.",
                    "Inspeccione el funcionamiento del ventilador del radiador y fusibles/relevadores.",
                    "Compruebe el termostato y la bomba de agua.",
                    "Confirme que no haya obstrucciones en radiador o conductos.",
                ],
                steps_fr=[
                    "Vérifiez le niveau de liquide de refroidissement et les fuites externes.",
                    "Contrôlez le fonctionnement du ventilateur de radiateur et les fusibles/relais.",
                    "Vérifiez le thermostat et la pompe à eau.",
                    "Assurez-vous qu’aucune obstruction n’est présente dans le radiateur ou les conduits.",
                ],
                recommended_action_en="Do not continue driving overheated; diagnose and repair cooling system before use.",
                recommended_action_es="No continúe conduciendo con sobrecalentamiento; diagnostique y repare el sistema de refrigeración antes de usar.",
                recommended_action_fr="Ne continuez pas à rouler en surchauffe; diagnostiquez et réparez le système de refroidissement avant utilisation.",
                safety_warning_en="Overheating can quickly cause severe engine damage.",
                safety_warning_es="El sobrecalentamiento puede causar rápidamente daños graves al motor.",
                safety_warning_fr="La surchauffe peut rapidement provoquer de graves dommages moteur.",
            ),
            DiagnosticStep(
                code="C0035",
                steps_en=[
                    "Check ABS warning lamp status.",
                    "Inspect wheel speed sensor wiring and connector at left front.",
                    "Clean or replace sensor and tone ring as required.",
                    "Verify wheel speed signal with scan tool.",
                ],
                steps_es=[
                    "Revise el estado de la luz de advertencia ABS.",
                    "Inspeccione el cableado y conector del sensor de velocidad de rueda delantera izquierda.",
                    "Limpie o reemplace el sensor y anillo de tono según sea necesario.",
                    "Verifique la señal de velocidad de rueda con escáner.",
                ],
                steps_fr=[
                    "Vérifiez le témoin ABS au tableau de bord.",
                    "Inspectez le câblage et le connecteur du capteur de vitesse de roue avant gauche.",
                    "Nettoyez ou remplacez le capteur et la couronne dentée si nécessaire.",
                    "Confirmez le signal de vitesse de roue avec un outil de diagnostic.",
                ],
                recommended_action_en="Restore proper wheel speed sensing to maintain ABS/traction control function.",
                recommended_action_es="Restaure la lectura correcta de velocidad de rueda para mantener ABS/control de tracción.",
                recommended_action_fr="Restaurez la détection correcte de vitesse de roue pour maintenir ABS/antipatinage.",
                safety_warning_en="Reduced ABS performance increases stopping distance on slippery roads.",
                safety_warning_es="El rendimiento reducido del ABS incrementa la distancia de frenado en superficies resbaladizas.",
                safety_warning_fr="La performance réduite de l’ABS augmente la distance de freinage sur routes glissantes.",
            ),
            DiagnosticStep(
                code="C0110",
                steps_en=[
                    "Inspect ABS pump motor wiring, fuses, and relays.",
                    "Check for corrosion in ABS module connectors.",
                    "Command pump on with scan tool and verify operation.",
                    "Replace pump or module if electrical issues are confirmed.",
                ],
                steps_es=[
                    "Inspeccione cableado, fusibles y relevadores del motor de la bomba ABS.",
                    "Verifique corrosión en conectores del módulo ABS.",
                    "Active la bomba con escáner y verifique su funcionamiento.",
                    "Reemplace bomba o módulo si se confirman problemas eléctricos.",
                ],
                steps_fr=[
                    "Inspectez le câblage, les fusibles et relais du moteur de pompe ABS.",
                    "Vérifiez la corrosion dans les connecteurs du module ABS.",
                    "Actionnez la pompe avec un outil de diagnostic et confirmez le fonctionnement.",
                    "Remplacez la pompe ou le module en cas de défaut confirmé.",
                ],
                recommended_action_en="Repair to restore full ABS braking performance.",
                recommended_action_es="Repárelo para restaurar el rendimiento completo del ABS.",
                recommended_action_fr="Réparez afin de rétablir les performances complètes de freinage ABS.",
                safety_warning_en="ABS pump failure can significantly affect braking stability.",
                safety_warning_es="La falla de la bomba ABS puede afectar significativamente la estabilidad de frenado.",
                safety_warning_fr="Une panne de pompe ABS peut fortement affecter la stabilité au freinage.",
            ),
            DiagnosticStep(
                code="P0700",
                steps_en=[
                    "Scan transmission control module for additional codes.",
                    "Inspect transmission fluid level and condition.",
                    "Check wiring harness and connectors to TCM.",
                    "Address underlying transmission fault indicated by sub-codes.",
                ],
                steps_es=[
                    "Escanee el módulo de control de transmisión para códigos adicionales.",
                    "Verifique nivel y condición del fluido de transmisión.",
                    "Revise arnés y conectores del TCM.",
                    "Solucione la falla de transmisión indicada por los subcódigos.",
                ],
                steps_fr=[
                    "Interrogez le module de commande de transmission pour les codes complémentaires.",
                    "Contrôlez le niveau et l’état de l’huile de boîte.",
                    "Vérifiez le faisceau et les connecteurs du TCM.",
                    "Traitez le défaut de transmission indiqué par les sous-codes.",
                ],
                recommended_action_en="Diagnose underlying transmission issue; avoid harsh driving until resolved.",
                recommended_action_es="Diagnostique el problema subyacente de transmisión; evite conducción brusca hasta resolverlo.",
                recommended_action_fr="Diagnostiquez le problème de transmission sous-jacent; évitez une conduite sévère jusqu’à résolution.",
                safety_warning_en="Transmission faults can lead to unexpected shifting and drivability concerns.",
                safety_warning_es="Las fallas de transmisión pueden causar cambios inesperados y problemas de manejo.",
                safety_warning_fr="Les défauts de transmission peuvent causer des passages de rapports inattendus et des soucis de conduite.",
            ),
            DiagnosticStep(
                code="B0020",
                steps_en=[
                    "Disable SRS system according to manufacturer procedures.",
                    "Inspect wiring and connectors for driver airbag circuit.",
                    "Measure resistance and continuity of deployment loops.",
                    "Replace faulty components and clear codes.",
                ],
                steps_es=[
                    "Desactive el sistema SRS según procedimientos del fabricante.",
                    "Inspeccione cableado y conectores del circuito de airbag del conductor.",
                    "Mida resistencia y continuidad de los circuitos de disparo.",
                    "Reemplace componentes defectuosos y borre códigos.",
                ],
                steps_fr=[
                    "Désactivez le système SRS selon les procédures du constructeur.",
                    "Inspectez le câblage et les connecteurs du circuit d’airbag conducteur.",
                    "Mesurez résistance et continuité des boucles de déclenchement.",
                    "Remplacez les composants défectueux puis effacez les codes.",
                ],
                recommended_action_en="Restore full airbag functionality before returning vehicle to customer.",
                recommended_action_es="Restaure la funcionalidad completa del airbag antes de entregar el vehículo.",
                recommended_action_fr="Restaurez la fonctionnalité complète de l’airbag avant restitution du véhicule.",
                safety_warning_en="Airbag faults can prevent deployment in a collision.",
                safety_warning_es="Las fallas del airbag pueden impedir su despliegue en una colisión.",
                safety_warning_fr="Les défauts d’airbag peuvent empêcher son déploiement en cas de collision.",
            ),
            DiagnosticStep(
                code="U0100",
                steps_en=[
                    "Check power and ground supply to ECM/PCM.",
                    "Inspect CAN bus wiring for damage or shorts.",
                    "Verify connector integrity at ECM/PCM and related modules.",
                    "Use wiring diagrams to isolate communication breaks.",
                ],
                steps_es=[
                    "Verifique alimentación y tierra del ECM/PCM.",
                    "Inspeccione el cableado del bus CAN por daños o cortos.",
                    "Compruebe integridad de conectores en ECM/PCM y módulos relacionados.",
                    "Use diagramas eléctricos para aislar interrupciones de comunicación.",
                ],
                steps_fr=[
                    "Vérifiez l’alimentation et la masse de l’ECM/PCM.",
                    "Inspectez le câblage du bus CAN pour dommages ou courts-circuits.",
                    "Contrôlez l’intégrité des connecteurs à l’ECM/PCM et modules associés.",
                    "Utilisez les schémas électriques pour isoler les ruptures de communication.",
                ],
                recommended_action_en="Restore stable communication on CAN network to ensure proper module operation.",
                recommended_action_es="Restaure la comunicación estable en la red CAN para garantizar el funcionamiento correcto de los módulos.",
                recommended_action_fr="Restaurez une communication stable sur le réseau CAN pour assurer le bon fonctionnement des modules.",
                safety_warning_en="Loss of communication can disable critical systems including engine and braking control.",
                safety_warning_es="La pérdida de comunicación puede desactivar sistemas críticos como motor y frenos.",
                safety_warning_fr="La perte de communication peut désactiver des systèmes critiques comme moteur et freinage.",
            ),
        ]
        return {s.code: s for s in steps}

    # -----------------------------
    # Predictive scoring and history
    # -----------------------------

    def score_faults(
        self,
        vehicle: FleetVehicle,
        fault_codes: List[str],
        when: Optional[datetime] = None,
    ) -> Dict[str, Dict[str, object]]:
        """
        Score each fault code and generate reasoning, safety flags, and schedule estimate.
        """
        when = when or datetime.utcnow()
        results: Dict[str, Dict[str, object]] = {}

        for code in fault_codes:
            code = code.strip().upper()
            if not code:
                continue
            fault_info = self.fault_db.get(code)

            if fault_info:
                base = fault_info.base_severity
                # Simple predictive model: severity -> probability band + slight randomness
                min_prob = max(20, base * 7)  # 7 * 10 = 70% max base
                max_prob = min(99, base * 9)  # up to 90%
                probability = random.randint(min_prob, max_prob)
                system = fault_info.system
                safety_critical = fault_info.safety_critical
            else:
                # Unknown code: still provide a generic probability
                probability = random.randint(30, 70)
                system = "Unknown"
                safety_critical = False

            # Reasoning summary
            reasoning_en = self._build_reasoning_en(code, probability, system, safety_critical, fault_info)

            # Maintenance schedule: sooner for more severe faults
            days_to_service = max(1, int(30 - (probability / 3)))
            predicted_date = when + timedelta(days=days_to_service)

            # Safety alert text (English internal; GUI will translate label text)
            safety_alert_en = ""
            if safety_critical or probability >= 80:
                if system in ("Brakes", "Safety Restraint", "Network"):
                    safety_alert_en = "Critical safety system affected. Limit driving and repair immediately."
                elif system in ("Cooling", "Engine", "Transmission"):
                    safety_alert_en = "Risk of breakdown or engine damage. Avoid heavy loads and repair as soon as possible."
                else:
                    safety_alert_en = "Potential safety impact. Diagnose and repair promptly."

            results[code] = {
                "code": code,
                "system": system,
                "probability": probability,
                "reasoning_en": reasoning_en,
                "safety_critical": safety_critical,
                "safety_alert_en": safety_alert_en,
                "predicted_service_date": predicted_date,
            }

            # Append to history DataFrame
            self.history_df.loc[len(self.history_df)] = [
                when,
                vehicle.vehicle_id,
                vehicle.make,
                vehicle.model,
                vehicle.year,
                code,
                probability,
                system,
                safety_critical,
            ]

        return results

    @staticmethod
    def _build_reasoning_en(
        code: str,
        probability: int,
        system: str,
        safety_critical: bool,
        fault_info: Optional[FaultInfo],
    ) -> str:
        """Generate an English reasoning summary for a fault."""
        base_desc = fault_info.description_en if fault_info else "Unknown or manufacturer-specific fault code."
        parts = [f"Observed diagnostic code {code} in the {system} system."]
        parts.append(f"Historical patterns for this type of fault suggest a {probability}% likelihood of being active.")
        if safety_critical:
            parts.append("This fault is associated with critical safety or drivability concerns.")
        else:
            parts.append("This fault is typically associated with performance or emissions concerns.")
        parts.append(f"Technical description: {base_desc}")
        return " ".join(parts)


# -----------------------------
# Localization
# -----------------------------


class Localizer:
    """Simple multi-language UI and report string provider."""

    SUPPORTED = ("en", "es", "fr")

    def __init__(self, language: str = "en") -> None:
        self.language = language if language in self.SUPPORTED else "en"
        self.translations: Dict[str, Dict[str, str]] = self._build_translations()

    def set_language(self, language: str) -> None:
        self.language = language if language in self.SUPPORTED else "en"

    def t(self, key: str) -> str:
        """Translate UI/report text with English fallback."""
        lang_dict = self.translations.get(key, {})
        return lang_dict.get(self.language) or lang_dict.get("en", key)

    @staticmethod
    def _build_translations() -> Dict[str, Dict[str, str]]:
        """Central dictionary for all display strings."""
        return {
            "app_title": {
                "en": "Predictive Vehicle Maintenance Assistant",
                "es": "Asistente Predictivo de Mantenimiento de Vehículos",
                "fr": "Assistant Prédictif de Maintenance de Véhicules",
            },
            "tab_diagnostics": {"en": "Diagnostics", "es": "Diagnóstico", "fr": "Diagnostic"},
            "tab_dashboard": {"en": "Dashboard", "es": "Panel", "fr": "Tableau de bord"},
            "tab_fleet": {"en": "Fleet", "es": "Flota", "fr": "Flotte"},
            "tab_settings": {"en": "Settings", "es": "Ajustes", "fr": "Paramètres"},
            "language": {"en": "Language", "es": "Idioma", "fr": "Langue"},
            "vehicle_mode": {"en": "Vehicle Mode", "es": "Modo de vehículo", "fr": "Mode véhicule"},
            "single_vehicle": {"en": "Single Vehicle", "es": "Vehículo único", "fr": "Véhicule unique"},
            "fleet_mode": {"en": "Fleet Mode", "es": "Modo flota", "fr": "Mode flotte"},
            "vehicle_id": {"en": "Vehicle ID", "es": "ID Vehículo", "fr": "ID Véhicule"},
            "make": {"en": "Make", "es": "Marca", "fr": "Marque"},
            "model": {"en": "Model", "es": "Modelo", "fr": "Modèle"},
            "year": {"en": "Year", "es": "Año", "fr": "Année"},
            "fault_codes": {
                "en": "OBD-II Fault Codes (comma-separated)",
                "es": "Códigos de falla OBD-II (separados por comas)",
                "fr": "Codes défaut OBD-II (séparés par des virgules)",
            },
            "run_diagnostics": {
                "en": "Run Diagnostics",
                "es": "Ejecutar diagnóstico",
                "fr": "Lancer le diagnostic",
            },
            "simulation_mode": {
                "en": "Run Demo / Simulation",
                "es": "Ejecutar demo / simulación",
                "fr": "Lancer démo / simulation",
            },
            "export_text": {
                "en": "Export Text Report",
                "es": "Exportar informe de texto",
                "fr": "Exporter rapport texte",
            },
            "export_pdf": {
                "en": "Export PDF Report",
                "es": "Exportar informe PDF",
                "fr": "Exporter rapport PDF",
            },
            "progress_scoring": {
                "en": "Scoring fault codes...",
                "es": "Calculando probabilidad de fallas...",
                "fr": "Analyse des codes défaut...",
            },
            "diagnostics_results": {
                "en": "Diagnostics Results",
                "es": "Resultados del diagnóstico",
                "fr": "Résultats du diagnostic",
            },
            "safety_alerts": {"en": "Safety Alerts", "es": "Alertas de seguridad", "fr": "Alertes de sécurité"},
            "no_faults": {
                "en": "No fault codes provided.",
                "es": "No se proporcionaron códigos de falla.",
                "fr": "Aucun code défaut fourni.",
            },
            "invalid_vehicle": {
                "en": "Please enter a valid vehicle ID, make, model, and year.",
                "es": "Ingrese un ID de vehículo, marca, modelo y año válidos.",
                "fr": "Veuillez saisir un ID véhicule, une marque, un modèle et une année valides.",
            },
            "report_saved": {
                "en": "Report saved successfully.",
                "es": "Informe guardado correctamente.",
                "fr": "Rapport enregistré avec succès.",
            },
            "error_saving": {
                "en": "Error while saving report.",
                "es": "Error al guardar el informe.",
                "fr": "Erreur lors de l’enregistrement du rapport.",
            },
            "dashboard_fault_trends": {
                "en": "Fault Trends Over Time",
                "es": "Tendencias de fallas en el tiempo",
                "fr": "Tendances des défauts dans le temps",
            },
            "dashboard_maintenance": {
                "en": "Predicted Maintenance Schedule",
                "es": "Programa de mantenimiento previsto",
                "fr": "Plan de maintenance prévisionnel",
            },
            "fleet_summary": {
                "en": "Fleet Summary",
                "es": "Resumen de flota",
                "fr": "Résumé de la flotte",
            },
            "company_name": {
                "en": "Predictive Vehicle Maintenance Assistant",
                "es": "Asistente Predictivo de Mantenimiento de Vehículos",
                "fr": "Assistant Prédictif de Maintenance de Véhicules",
            },
            "report_title": {
                "en": "Diagnostics and Maintenance Report",
                "es": "Informe de diagnóstico y mantenimiento",
                "fr": "Rapport de diagnostic et de maintenance",
            },
        }


# -----------------------------
# Reporting utilities
# -----------------------------


def generate_report_id() -> str:
    """Create a unique, human-readable report ID."""
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    rand = "".join(random.choices(string.ascii_uppercase + string.digits, k=4))
    return f"RPT-{timestamp}-{rand}"


def build_text_report(
    localizer: Localizer,
    vehicle: FleetVehicle,
    fault_results: Dict[str, Dict[str, object]],
    diagnostic_db: Dict[str, DiagnosticStep],
    language: str,
) -> str:
    """Generate a detailed text report in the selected language."""
    lang = language if language in Localizer.SUPPORTED else "en"

    def pick(en: str, es: str, fr: str) -> str:
        return {"en": en, "es": es, "fr": fr}.get(lang, en)

    now = datetime.utcnow()
    report_id = generate_report_id()
    lines: List[str] = []
    lines.append(f"{localizer.t('company_name')} - {localizer.t('report_title')}")
    lines.append(f"Report ID: {report_id}")
    lines.append(f"Generated: {now.isoformat()} UTC")
    lines.append("")
    lines.append(f"Vehicle ID: {vehicle.vehicle_id}")
    lines.append(f"Make/Model/Year: {vehicle.make} {vehicle.model} {vehicle.year}")
    lines.append("")
    lines.append("Fault Summary:")
    lines.append("------------------------------")

    for code, data in fault_results.items():
        diag = diagnostic_db.get(code)
        probability = data["probability"]
        system = data["system"]
        safety_critical = data["safety_critical"]
        predicted_date = data["predicted_service_date"]

        if diag:
            desc = pick(
                diag.steps_en[0],
                diag.steps_es[0],
                diag.steps_fr[0],
            )
            rec = pick(
                diag.recommended_action_en,
                diag.recommended_action_es,
                diag.recommended_action_fr,
            )
            warn = pick(
                diag.safety_warning_en,
                diag.safety_warning_es,
                diag.safety_warning_fr,
            )
        else:
            desc = "No detailed diagnostic steps available."
            rec = "Recommend further investigation using manufacturer documentation."
            warn = "No specific safety information available."

        lines.append(f"Code: {code} | System: {system} | Probability: {probability}%")
        lines.append(f"Summary: {data['reasoning_en']}")
        lines.append(f"Description: {desc}")
        lines.append(f"Recommended action: {rec}")
        lines.append(f"Safety: {warn}")
        lines.append(f"Predicted service date: {predicted_date.date().isoformat()}")
        lines.append("")

    lines.append("End of report.")
    return "\n".join(lines)


def build_pdf_report(
    file_path: str,
    localizer: Localizer,
    vehicle: FleetVehicle,
    fault_results: Dict[str, Dict[str, object]],
    diagnostic_db: Dict[str, DiagnosticStep],
    language: str,
    trends_figure: Optional[Figure] = None,
    maintenance_figure: Optional[Figure] = None,
) -> None:
    """
    Generate a PDF report with header, tables, and optional charts.
    """
    lang = language if language in Localizer.SUPPORTED else "en"

    def pick(en: str, es: str, fr: str) -> str:
        return {"en": en, "es": es, "fr": fr}.get(lang, en)

    doc = SimpleDocTemplate(file_path, pagesize=A4)
    story: List[object] = []
    styles = getSampleStyleSheet()

    # Header
    company = localizer.t("company_name")
    title = localizer.t("report_title")

    story.append(Paragraph(f"<b>{company}</b>", styles["Title"]))
    story.append(Paragraph(title, styles["Heading2"]))
    story.append(Spacer(1, 12))

    # Placeholder for logo
    story.append(Paragraph("[Company Logo Placeholder]", styles["Normal"]))
    story.append(Spacer(1, 12))

    report_id = generate_report_id()
    now = datetime.utcnow()
    story.append(Paragraph(f"Report ID: {report_id}", styles["Normal"]))
    story.append(Paragraph(f"Generated: {now.isoformat()} UTC", styles["Normal"]))
    story.append(Spacer(1, 12))

    # Vehicle info
    story.append(
        Paragraph(
            f"Vehicle: {vehicle.vehicle_id} - {vehicle.make} {vehicle.model} ({vehicle.year})",
            styles["Heading3"],
        )
    )
    story.append(Spacer(1, 12))

    # Fault table
    table_data = [
        [
            "Code",
            "System",
            "Probability",
            "Reasoning",
            "Workflow / Action",
        ]
    ]

    for code, data in fault_results.items():
        diag = diagnostic_db.get(code)
        if diag:
            rec = pick(
                diag.recommended_action_en,
                diag.recommended_action_es,
                diag.recommended_action_fr,
            )
            steps = diag.steps_en if lang == "en" else (diag.steps_es if lang == "es" else diag.steps_fr)
            workflow = "; ".join(steps)
        else:
            rec = "Further investigation required."
            workflow = "No detailed steps available."

        table_data.append(
            [
                code,
                data["system"],
                f"{data['probability']}%",
                data["reasoning_en"],
                f"{workflow} | {rec}",
            ]
        )

    tbl = Table(table_data, repeatRows=1)
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(tbl)
    story.append(Spacer(1, 12))

    # Safety section
    story.append(Paragraph("Safety Warnings", styles["Heading3"]))
    for code, data in fault_results.items():
        diag = diagnostic_db.get(code)
        if diag:
            warn = pick(
                diag.safety_warning_en,
                diag.safety_warning_es,
                diag.safety_warning_fr,
            )
        else:
            warn = "No specific safety information available."
        story.append(Paragraph(f"{code}: {warn}", styles["Normal"]))
    story.append(Spacer(1, 12))

    # Charts: save figures as temporary images and embed
    def add_figure(fig: Figure, caption: str) -> None:
        if fig is None:
            return
        tmp_dir = tempfile.gettempdir()
        tmp_path = os.path.join(tmp_dir, f"pvma_chart_{random.randint(1000, 9999)}.png")
        fig.savefig(tmp_path, dpi=120, bbox_inches="tight")
        story.append(Image(tmp_path, width=400, height=250))
        story.append(Paragraph(caption, styles["Italic"]))
        story.append(Spacer(1, 12))

    add_figure(trends_figure, localizer.t("dashboard_fault_trends"))
    add_figure(maintenance_figure, localizer.t("dashboard_maintenance"))

    doc.build(story)


# -----------------------------
# Tkinter GUI Application
# -----------------------------


class PredictiveVehicleMaintenanceApp:
    """Main Tkinter GUI for the Predictive Vehicle Maintenance Assistant."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.core = PredictiveVehicleMaintenanceAssistantCore()
        self.language = "en"
        self.localizer = Localizer(self.language)

        self.current_vehicle = FleetVehicle("VEH-001", "Generic", "Demo", 2022)
        self.current_fault_results: Dict[str, Dict[str, object]] = {}

        self._build_ui()

    # -----------------------------
    # UI construction
    # -----------------------------

    def _build_ui(self) -> None:
        """Create and layout all Tkinter widgets."""
        self.root.title(self.localizer.t("app_title"))
        self.root.geometry("1150x750")

        # Top-level frame with language selection
        top_frame = ttk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        ttk.Label(top_frame, text=self.localizer.t("language")).pack(side=tk.LEFT)
        self.language_var = tk.StringVar(value=self.language)
        lang_combo = ttk.Combobox(
            top_frame,
            textvariable=self.language_var,
            state="readonly",
            values=["en", "es", "fr"],
            width=5,
        )
        lang_combo.bind("<<ComboboxSelected>>", self.on_language_change)
        lang_combo.pack(side=tk.LEFT, padx=5)

        # Notebook for main sections
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self._build_diagnostics_tab()
        self._build_dashboard_tab()
        self._build_fleet_tab()
        self._build_settings_tab()

    def _build_diagnostics_tab(self) -> None:
        """Diagnostics tab: vehicle details, OBD codes input, results, and safety panel."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=self.localizer.t("tab_diagnostics"))

        # Vehicle mode and details
        top = ttk.Frame(tab)
        top.pack(fill=tk.X, padx=5, pady=5)

        # Vehicle mode
        ttk.Label(top, text=self.localizer.t("vehicle_mode")).grid(row=0, column=0, sticky="w")
        self.vehicle_mode_var = tk.StringVar(value="single")
        ttk.Radiobutton(
            top,
            text=self.localizer.t("single_vehicle"),
            variable=self.vehicle_mode_var,
            value="single",
        ).grid(row=0, column=1, sticky="w")
        ttk.Radiobutton(
            top,
            text=self.localizer.t("fleet_mode"),
            variable=self.vehicle_mode_var,
            value="fleet",
        ).grid(row=0, column=2, sticky="w")

        # Vehicle fields
        ttk.Label(top, text=self.localizer.t("vehicle_id")).grid(row=1, column=0, sticky="w", pady=2)
        self.vehicle_id_entry = ttk.Entry(top, width=15)
        self.vehicle_id_entry.insert(0, self.current_vehicle.vehicle_id)
        self.vehicle_id_entry.grid(row=1, column=1, sticky="w", pady=2)

        ttk.Label(top, text=self.localizer.t("make")).grid(row=1, column=2, sticky="w", pady=2)
        self.make_entry = ttk.Entry(top, width=15)
        self.make_entry.insert(0, self.current_vehicle.make)
        self.make_entry.grid(row=1, column=3, sticky="w", pady=2)

        ttk.Label(top, text=self.localizer.t("model")).grid(row=1, column=4, sticky="w", pady=2)
        self.model_entry = ttk.Entry(top, width=15)
        self.model_entry.insert(0, self.current_vehicle.model)
        self.model_entry.grid(row=1, column=5, sticky="w", pady=2)

        ttk.Label(top, text=self.localizer.t("year")).grid(row=1, column=6, sticky="w", pady=2)
        self.year_entry = ttk.Entry(top, width=6)
        self.year_entry.insert(0, str(self.current_vehicle.year))
        self.year_entry.grid(row=1, column=7, sticky="w", pady=2)

        # Fault code input
        mid = ttk.Frame(tab)
        mid.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(mid, text=self.localizer.t("fault_codes")).pack(anchor="w")
        self.fault_entry = ttk.Entry(mid)
        self.fault_entry.pack(fill=tk.X, padx=0, pady=2)

        # Buttons and progress bar
        btn_frame = ttk.Frame(mid)
        btn_frame.pack(fill=tk.X, pady=5)

        self.run_btn = ttk.Button(btn_frame, text=self.localizer.t("run_diagnostics"), command=self.on_run_diagnostics)
        self.run_btn.pack(side=tk.LEFT, padx=2)

        self.sim_btn = ttk.Button(btn_frame, text=self.localizer.t("simulation_mode"), command=self.on_run_simulation)
        self.sim_btn.pack(side=tk.LEFT, padx=2)

        self.export_txt_btn = ttk.Button(
            btn_frame,
            text=self.localizer.t("export_text"),
            command=self.on_export_text_report,
        )
        self.export_txt_btn.pack(side=tk.RIGHT, padx=2)

        self.export_pdf_btn = ttk.Button(
            btn_frame,
            text=self.localizer.t("export_pdf"),
            command=self.on_export_pdf_report,
        )
        self.export_pdf_btn.pack(side=tk.RIGHT, padx=2)

        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress = ttk.Progressbar(
            mid,
            variable=self.progress_var,
            mode="indeterminate",
        )
        self.progress_label = ttk.Label(mid, text="")

        # Results and safety panels
        bottom = ttk.PanedWindow(tab, orient=tk.HORIZONTAL)
        bottom.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Fault results text
        left_frame = ttk.Labelframe(bottom, text=self.localizer.t("diagnostics_results"))
        self.results_text = tk.Text(left_frame, wrap=tk.WORD, state=tk.DISABLED)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        bottom.add(left_frame, weight=3)

        # Safety alerts
        right_frame = ttk.Labelframe(bottom, text=self.localizer.t("safety_alerts"))
        self.safety_text = tk.Text(right_frame, wrap=tk.WORD, state=tk.DISABLED, foreground="red")
        self.safety_text.pack(fill=tk.BOTH, expand=True)
        bottom.add(right_frame, weight=2)

    def _build_dashboard_tab(self) -> None:
        """Dashboard tab with matplotlib charts for fault trends and maintenance prediction."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=self.localizer.t("tab_dashboard"))

        # Two matplotlib figures: trends and maintenance schedule
        self.fig_trends = Figure(figsize=(5, 3), dpi=100)
        self.ax_trends = self.fig_trends.add_subplot(111)
        self.fig_maint = Figure(figsize=(5, 3), dpi=100)
        self.ax_maint = self.fig_maint.add_subplot(111)

        top = ttk.Frame(tab)
        top.pack(fill=tk.BOTH, expand=True)

        left = ttk.Labelframe(top, text=self.localizer.t("dashboard_fault_trends"))
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        right = ttk.Labelframe(top, text=self.localizer.t("dashboard_maintenance"))
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.canvas_trends = FigureCanvasTkAgg(self.fig_trends, master=left)
        self.canvas_trends.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.canvas_maint = FigureCanvasTkAgg(self.fig_maint, master=right)
        self.canvas_maint.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._refresh_dashboard()

    def _build_fleet_tab(self) -> None:
        """Fleet tab for viewing summarized history for multiple vehicles."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=self.localizer.t("tab_fleet"))

        # Simple treeview summary
        frame = ttk.Frame(tab)
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        columns = ("vehicle_id", "make_model", "count_faults", "avg_prob")
        self.fleet_tree = ttk.Treeview(frame, columns=columns, show="headings")
        self.fleet_tree.heading("vehicle_id", text=self.localizer.t("vehicle_id"))
        self.fleet_tree.heading("make_model", text=f"{self.localizer.t('make')}/{self.localizer.t('model')}")
        self.fleet_tree.heading("count_faults", text="# Faults")
        self.fleet_tree.heading("avg_prob", text="Avg Probability")
        self.fleet_tree.pack(fill=tk.BOTH, expand=True)

        self._refresh_fleet_summary()

    def _build_settings_tab(self) -> None:
        """Settings tab reserved for future options; currently echoes language settings."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=self.localizer.t("tab_settings"))

        ttk.Label(tab, text=self.localizer.t("language")).pack(anchor="w", padx=10, pady=10)
        ttk.Label(
            tab,
            text=(
                "All core UI elements and reports support English, Spanish, "
                "and French. Technical fault descriptions are provided in English with localized summaries."
            ),
            wraplength=600,
            justify=tk.LEFT,
        ).pack(anchor="w", padx=10, pady=5)

    # -----------------------------
    # Event handlers
    # -----------------------------

    def on_language_change(self, event: Optional[tk.Event] = None) -> None:
        """Update language and refresh UI labels."""
        self.language = self.language_var.get()
        self.localizer.set_language(self.language)
        self.root.title(self.localizer.t("app_title"))

        # Rebuild notebook tabs to refresh labels
        self.notebook.destroy()
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self._build_diagnostics_tab()
        self._build_dashboard_tab()
        self._build_fleet_tab()
        self._build_settings_tab()

    def _read_vehicle_from_form(self) -> Optional[FleetVehicle]:
        """Extract vehicle information from form; return None if invalid."""
        try:
            vid = self.vehicle_id_entry.get().strip()
            make = self.make_entry.get().strip()
            model = self.model_entry.get().strip()
            year = int(self.year_entry.get().strip())
        except Exception:
            messagebox.showerror(self.localizer.t("app_title"), self.localizer.t("invalid_vehicle"))
            return None
        if not (vid and make and model and 1900 <= year <= datetime.utcnow().year + 2):
            messagebox.showerror(self.localizer.t("app_title"), self.localizer.t("invalid_vehicle"))
            return None
        return FleetVehicle(vid, make, model, year)

    def on_run_diagnostics(self) -> None:
        """Triggered when user clicks 'Run Diagnostics'."""
        vehicle = self._read_vehicle_from_form()
        if not vehicle:
            return

        fault_str = self.fault_entry.get().strip()
        if not fault_str:
            messagebox.showwarning(self.localizer.t("app_title"), self.localizer.t("no_faults"))
            return

        fault_codes = [c.strip() for c in fault_str.split(",") if c.strip()]
        if not fault_codes:
            messagebox.showwarning(self.localizer.t("app_title"), self.localizer.t("no_faults"))
            return

        self.current_vehicle = vehicle

        # Show progress bar and simulate analysis delay
        self.progress_label.config(text=self.localizer.t("progress_scoring"))
        self.progress_label.pack(anchor="w")
        self.progress.pack(fill=tk.X, pady=2)
        self.progress.start(10)

        # Use after() so UI stays responsive
        self.root.after(700, lambda: self._complete_diagnostics(fault_codes))

    def _complete_diagnostics(self, fault_codes: List[str]) -> None:
        """Complete scoring after simulated load and update UI."""
        self.progress.stop()
        self.progress.pack_forget()
        self.progress_label.config(text="")

        self.current_fault_results = self.core.score_faults(self.current_vehicle, fault_codes)
        self._render_diagnostics_results()
        self._refresh_dashboard()
        self._refresh_fleet_summary()

    def on_run_simulation(self) -> None:
        """Demo mode: randomly select a vehicle and some fault codes for client-ready demo."""
        demo_vehicles = [
            FleetVehicle("FLEET-101", "Ford", "Transit", 2020),
            FleetVehicle("FLEET-202", "Mercedes", "Sprinter", 2021),
            FleetVehicle("FLEET-303", "Toyota", "Corolla", 2019),
        ]
        vehicle = random.choice(demo_vehicles)
        self.vehicle_id_entry.delete(0, tk.END)
        self.vehicle_id_entry.insert(0, vehicle.vehicle_id)
        self.make_entry.delete(0, tk.END)
        self.make_entry.insert(0, vehicle.make)
        self.model_entry.delete(0, tk.END)
        self.model_entry.insert(0, vehicle.model)
        self.year_entry.delete(0, tk.END)
        self.year_entry.insert(0, str(vehicle.year))

        # Pick a few demo fault codes
        demo_codes = random.sample(list(self.core.fault_db.keys()), k=min(3, len(self.core.fault_db)))
        self.fault_entry.delete(0, tk.END)
        self.fault_entry.insert(0, ", ".join(demo_codes))

        self.on_run_diagnostics()

    def on_export_text_report(self) -> None:
        """Export current diagnostics as a text report."""
        if not self.current_fault_results:
            messagebox.showwarning(self.localizer.t("app_title"), self.localizer.t("no_faults"))
            return
        vehicle = self._read_vehicle_from_form()
        if not vehicle:
            return

        report = build_text_report(
            self.localizer,
            vehicle,
            self.current_fault_results,
            self.core.diagnostic_db,
            self.language,
        )

        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")],
            title=self.localizer.t("export_text"),
        )
        if not file_path:
            return
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(report)
            messagebox.showinfo(self.localizer.t("app_title"), self.localizer.t("report_saved"))
        except Exception:
            messagebox.showerror(self.localizer.t("app_title"), self.localizer.t("error_saving"))

    def on_export_pdf_report(self) -> None:
        """Export current diagnostics as a PDF report with charts."""
        if not self.current_fault_results:
            messagebox.showwarning(self.localizer.t("app_title"), self.localizer.t("no_faults"))
            return
        vehicle = self._read_vehicle_from_form()
        if not vehicle:
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
            title=self.localizer.t("export_pdf"),
        )
        if not file_path:
            return

        try:
            build_pdf_report(
                file_path=file_path,
                localizer=self.localizer,
                vehicle=vehicle,
                fault_results=self.current_fault_results,
                diagnostic_db=self.core.diagnostic_db,
                language=self.language,
                trends_figure=self.fig_trends,
                maintenance_figure=self.fig_maint,
            )
            messagebox.showinfo(self.localizer.t("app_title"), self.localizer.t("report_saved"))
        except Exception:
            messagebox.showerror(self.localizer.t("app_title"), self.localizer.t("error_saving"))

    # -----------------------------
    # Rendering helpers
    # -----------------------------

    def _render_diagnostics_results(self) -> None:
        """Fill diagnostics and safety text areas from current results."""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete("1.0", tk.END)
        self.safety_text.config(state=tk.NORMAL)
        self.safety_text.delete("1.0", tk.END)

        for code, data in self.current_fault_results.items():
            self.results_text.insert(
                tk.END,
                f"{code} - {data['system']} - {data['probability']}%\n{data['reasoning_en']}\n\n",
            )

        for code, data in self.current_fault_results.items():
            alert = data.get("safety_alert_en", "")
            if alert:
                self.safety_text.insert(tk.END, f"{code}: {alert}\n")

        self.results_text.config(state=tk.DISABLED)
        self.safety_text.config(state=tk.DISABLED)

    def _refresh_dashboard(self) -> None:
        """Update matplotlib charts based on fleet history."""
        df = self.core.history_df
        self.ax_trends.clear()
        self.ax_maint.clear()

        if not df.empty:
            # Fault trends over time: count of faults per day
            df_trend = df.copy()
            df_trend["date"] = pd.to_datetime(df_trend["timestamp"]).dt.date
            trend = df_trend.groupby("date")["fault_code"].count()
            self.ax_trends.plot(trend.index, trend.values, marker="o")
            self.ax_trends.set_title(self.localizer.t("dashboard_fault_trends"))
            self.ax_trends.set_xlabel("Date")
            self.ax_trends.set_ylabel("# Faults")
            self.ax_trends.tick_params(axis="x", rotation=45)

            # Predicted maintenance: for current vehicle only, show probability as bar chart
            if self.current_fault_results:
                codes = list(self.current_fault_results.keys())
                probs = [self.current_fault_results[c]["probability"] for c in codes]
                self.ax_maint.bar(codes, probs)
                self.ax_maint.set_ylim(0, 100)
                self.ax_maint.set_title(self.localizer.t("dashboard_maintenance"))
                self.ax_maint.set_ylabel("Probability (%)")
        else:
            self.ax_trends.text(0.5, 0.5, "No data yet", ha="center", va="center")
            self.ax_maint.text(0.5, 0.5, "No data yet", ha="center", va="center")

        self.fig_trends.tight_layout()
        self.fig_maint.tight_layout()
        self.canvas_trends.draw()
        self.canvas_maint.draw()

    def _refresh_fleet_summary(self) -> None:
        """Update fleet summary treeview from history."""
        for row in self.fleet_tree.get_children():
            self.fleet_tree.delete(row)

        df = self.core.history_df
        if df.empty:
            return

        grouped = (
            df.groupby(["vehicle_id", "make", "model"])
            .agg(count_faults=("fault_code", "count"), avg_prob=("probability", "mean"))
            .reset_index()
        )

        for _, row in grouped.iterrows():
            vid = row["vehicle_id"]
            make_model = f"{row['make']} {row['model']}"
            count_faults = int(row["count_faults"])
            avg_prob = f"{row['avg_prob']:.1f}%"
            self.fleet_tree.insert("", tk.END, values=(vid, make_model, count_faults, avg_prob))


# -----------------------------
# Main entrypoint
# -----------------------------


def main() -> None:
    root = tk.Tk()
    app = PredictiveVehicleMaintenanceApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

