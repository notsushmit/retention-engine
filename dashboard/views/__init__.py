"""One module per page; each has render(client, health, capture)."""

from dashboard.views import action_planner, at_risk_list, customer_profile, model_health, overview

VIEWS = {
    "overview": overview,
    "at_risk_list": at_risk_list,
    "customer_profile": customer_profile,
    "action_planner": action_planner,
    "model_health": model_health,
}
