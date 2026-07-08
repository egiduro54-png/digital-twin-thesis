"""
suitability.py

MiFID II Suitability Assessment Engine.

Implements a structured investor questionnaire covering the five MiFID II
suitability dimensions:
  1. Investment knowledge & experience
  2. Investment objectives
  3. Investment horizon
  4. Financial situation
  5. Risk tolerance

Each answer carries a score. The total score maps to a risk profile
(liquidity_plus → dynamic) and a PRIIPs SRI band (1-7).
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class SuitabilityResult:
    score: int
    max_score: int
    risk_profile: str
    sri_score: int
    profile_label: str
    dimension_scores: dict[str, int] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Questionnaire definition
# ---------------------------------------------------------------------------

QUESTIONS: list[dict] = [
    # ── Dimension 1: Knowledge & Experience ─────────────────────────────
    {
        "id": "knowledge_products",
        "dimension": "Γνώση & Εμπειρία",
        "text": "Με ποια χρηματοοικονομικά προϊόντα έχετε εμπειρία;",
        "options": [
            ("Καταθέσεις / Ταμιευτήριο μόνο", 0),
            ("Αμοιβαία κεφάλαια / ETFs", 2),
            ("Μετοχές εισηγμένων εταιρειών", 3),
            ("Παράγωγα, CFDs, δομημένα προϊόντα", 4),
        ],
    },
    {
        "id": "knowledge_years",
        "dimension": "Γνώση & Εμπειρία",
        "text": "Πόσα χρόνια επενδυτικής εμπειρίας έχετε;",
        "options": [
            ("Καμία εμπειρία", 0),
            ("Λιγότερο από 2 χρόνια", 1),
            ("2-5 χρόνια", 2),
            ("Περισσότερα από 5 χρόνια", 3),
        ],
    },
    # ── Dimension 2: Investment Objectives ──────────────────────────────
    {
        "id": "objective",
        "dimension": "Επενδυτικοί Στόχοι",
        "text": "Ποιος είναι ο κύριος επενδυτικός σας στόχος;",
        "options": [
            ("Διατήρηση κεφαλαίου — ελάχιστος κίνδυνος", 0),
            ("Σταθερό εισόδημα με χαμηλό κίνδυνο", 1),
            ("Ισορροπία μεταξύ εισοδήματος και ανάπτυξης", 2),
            ("Μακροπρόθεσμη ανάπτυξη κεφαλαίου", 3),
            ("Μεγιστοποίηση απόδοσης — αποδέχομαι υψηλό κίνδυνο", 4),
        ],
    },
    # ── Dimension 3: Investment Horizon ─────────────────────────────────
    {
        "id": "horizon",
        "dimension": "Επενδυτικός Ορίζοντας",
        "text": "Για πόσο χρονικό διάστημα σκοπεύετε να επενδύσετε;",
        "options": [
            ("Λιγότερο από 1 χρόνο", 0),
            ("1-3 χρόνια", 1),
            ("3-5 χρόνια", 2),
            ("5-10 χρόνια", 3),
            ("Περισσότερα από 10 χρόνια", 4),
        ],
    },
    {
        "id": "liquidity_need",
        "dimension": "Επενδυτικός Ορίζοντας",
        "text": "Πόσο γρήγορα μπορεί να χρειαστείτε πρόσβαση στα κεφάλαιά σας;",
        "options": [
            ("Άμεσα (εντός 1 μήνα)", 0),
            ("Εντός 6 μηνών", 1),
            ("Εντός 1-3 ετών", 2),
            ("Δεν έχω ανάγκη σύντομης ρευστοποίησης", 3),
        ],
    },
    # ── Dimension 4: Financial Situation ────────────────────────────────
    {
        "id": "income_stability",
        "dimension": "Οικονομική Κατάσταση",
        "text": "Πώς χαρακτηρίζετε το εισόδημά σας;",
        "options": [
            ("Ασταθές / αβέβαιο", 0),
            ("Σταθερό αλλά με λίγα πλεονάσματα", 1),
            ("Σταθερό με κάποια πλεονάσματα", 2),
            ("Υψηλό και σταθερό", 3),
        ],
    },
    {
        "id": "loss_capacity",
        "dimension": "Οικονομική Κατάσταση",
        "text": "Τι ποσοστό της επένδυσής σας μπορείτε να χάσετε χωρίς να επηρεαστεί η ποιότητα ζωής σας;",
        "options": [
            ("Κανένα — δεν αντέχω καμία ζημιά", 0),
            ("Έως 5%", 1),
            ("5-15%", 2),
            ("15-30%", 3),
            ("Περισσότερο από 30%", 4),
        ],
    },
    # ── Dimension 5: Risk Tolerance ──────────────────────────────────────
    {
        "id": "risk_reaction",
        "dimension": "Ανοχή Κινδύνου",
        "text": "Αν η επένδυσή σας έχασε 20% της αξίας της σε 1 μήνα, τι θα κάνατε;",
        "options": [
            ("Θα πουλούσα αμέσως για να αποφύγω μεγαλύτερες ζημιές", 0),
            ("Θα πουλούσα ένα μέρος", 1),
            ("Θα περίμενα χωρίς να κάνω τίποτα", 2),
            ("Θα αγόραζα περισσότερο εκμεταλλευόμενος τη χαμηλή τιμή", 3),
        ],
    },
    {
        "id": "risk_return",
        "dimension": "Ανοχή Κινδύνου",
        "text": "Ποια σχέση απόδοσης/κινδύνου προτιμάτε;",
        "options": [
            ("Εγγυημένη μηδενική ζημιά, χαμηλή απόδοση", 0),
            ("Χαμηλός κίνδυνος ζημιάς, μέτρια απόδοση", 1),
            ("Μέτριος κίνδυνος, ικανοποιητική απόδοση", 2),
            ("Υψηλός κίνδυνος, υψηλή πιθανή απόδοση", 3),
        ],
    },
]

# Score → risk profile mapping
SCORE_TO_PROFILE = [
    (0,  6,  "liquidity_plus", 1, "Liquidity Plus (Πολύ Χαμηλός Κίνδυνος)"),
    (7,  12, "defensive",      2, "Defensive (Χαμηλός Κίνδυνος)"),
    (13, 18, "flexible",       3, "Flexible (Μέτριος Κίνδυνος)"),
    (19, 24, "growth",         5, "Growth (Υψηλός Κίνδυνος)"),
    (25, 99, "dynamic",        6, "Dynamic (Πολύ Υψηλός Κίνδυνος)"),
]


def calculate_suitability(answers: dict[str, int]) -> SuitabilityResult:
    """
    Calculate suitability result from a dict of {question_id: score_value}.
    Returns a SuitabilityResult with profile, SRI score, and flags.
    """
    dimension_scores: dict[str, int] = {}
    total = 0

    for q in QUESTIONS:
        score = answers.get(q["id"], 0)
        dim = q["dimension"]
        dimension_scores[dim] = dimension_scores.get(dim, 0) + score
        total += score

    max_score = sum(max(s for _, s in q["options"]) for q in QUESTIONS)

    # Map to profile
    profile = "flexible"
    sri = 3
    label = "Flexible (Μέτριος Κίνδυνος)"
    for lo, hi, p, s, lbl in SCORE_TO_PROFILE:
        if lo <= total <= hi:
            profile, sri, label = p, s, lbl
            break

    # Flags for inconsistencies
    flags = []
    if answers.get("horizon", 0) <= 1 and answers.get("risk_reaction", 0) >= 2:
        flags.append("⚠️ Σύντομος ορίζοντας αλλά υψηλή ανοχή κινδύνου — επαληθεύστε με τον πελάτη.")
    if answers.get("loss_capacity", 0) == 0 and answers.get("objective", 0) >= 3:
        flags.append("⚠️ Μηδενική ανοχή ζημιάς αλλά υψηλός επενδυτικός στόχος — ασυμβατότητα.")
    if answers.get("knowledge_products", 0) == 0 and answers.get("objective", 0) >= 3:
        flags.append("⚠️ Περιορισμένη γνώση για τον επιλεγμένο επενδυτικό στόχο.")

    return SuitabilityResult(
        score=total,
        max_score=max_score,
        risk_profile=profile,
        sri_score=sri,
        profile_label=label,
        dimension_scores=dimension_scores,
        flags=flags,
    )
