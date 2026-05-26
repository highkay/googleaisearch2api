async function refreshSummary() {
    const response = await fetch("/console/summary.json", {
        headers: { Accept: "application/json" },
        cache: "no-store",
    });
    if (!response.ok) {
        return;
    }

    const payload = await response.json();
    const summary = payload.summary || {};
    const pool = payload.pool || {};
    const proxySessions = payload.proxy_sessions || {};
    const statMap = {
        total_requests: summary.total_requests,
        successful_requests: summary.successful_requests,
        failed_requests: summary.failed_requests,
        average_latency_ms: summary.average_latency_ms == null
            ? "n/a"
            : `${Math.round(summary.average_latency_ms)} ms`,
        busy_workers: pool.busy_workers,
        queued_requests: pool.queued_requests,
        workers_with_errors: pool.workers_with_errors,
        generation: pool.generation,
        sticky_active: proxySessions.active || 0,
        sticky_cooldown: proxySessions.cooldown || 0,
    };

    Object.entries(statMap).forEach(([key, value]) => {
        const element = document.querySelector(`[data-stat="${key}"]`);
        if (element) {
            element.textContent = value;
        }
    });
}

window.addEventListener("load", () => {
    setInterval(() => {
        refreshSummary().catch(() => {});
    }, 15000);
});
