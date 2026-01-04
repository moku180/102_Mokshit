// Fraud Detection Dashboard JavaScript

const API_BASE = 'http://localhost:5000/api';
let distributionChart, timelineChart;

// Initialize dashboard
document.addEventListener('DOMContentLoaded', () => {
    loadMetrics();
    loadAlerts();
    loadCharts();

    // Auto-refresh every 5 seconds
    setInterval(() => {
        loadMetrics();
        loadAlerts();
    }, 5000);

    // Filter change handler
    document.getElementById('riskFilter').addEventListener('change', loadAlerts);
});

// Load dashboard metrics
async function loadMetrics() {
    try {
        const response = await fetch(`${API_BASE}/metrics`);
        const data = await response.json();

        document.getElementById('totalAlerts').textContent = data.total_alerts.toLocaleString();
        document.getElementById('precision').textContent = (data.precision_at_100 * 100).toFixed(1) + '%';
        document.getElementById('avgScore').textContent = data.avg_fraud_score.toFixed(3);
        document.getElementById('trueFrauds').textContent = data.true_frauds_caught.toLocaleString();
    } catch (error) {
        console.error('Error loading metrics:', error);
    }
}

// Load fraud alerts
async function loadAlerts() {
    const threshold = document.getElementById('riskFilter').value;
    const container = document.getElementById('alertsContainer');

    try {
        const response = await fetch(`${API_BASE}/alerts?threshold=${threshold}&limit=50`);
        const data = await response.json();

        if (data.alerts.length === 0) {
            container.innerHTML = '<div class="loading">No alerts found</div>';
            return;
        }

        container.innerHTML = data.alerts.map(alert => `
            <div class="alert-item ${alert.risk_level}" onclick="showExplanation(${alert.transaction_id})">
                <div class="alert-header">
                    <span class="alert-id">Transaction #${alert.transaction_id}</span>
                    <span class="risk-badge ${alert.risk_level}">${alert.risk_level}</span>
                </div>
                <div class="alert-score">${(alert.fraud_score * 100).toFixed(1)}%</div>
                <div class="alert-details">
                    <span>Rank: #${alert.rank}</span>
                    <span>Actual: ${alert.actual_fraud ? '✅ Fraud' : '❌ Legit'}</span>
                </div>
            </div>
        `).join('');
    } catch (error) {
        console.error('Error loading alerts:', error);
        container.innerHTML = '<div class="loading">Error loading alerts</div>';
    }
}

// Show SHAP explanation modal
async function showExplanation(transactionId) {
    console.log('Showing explanation for transaction:', transactionId);
    const modal = document.getElementById('shapModal');
    const content = document.getElementById('shapContent');

    modal.classList.add('active');
    content.innerHTML = '<div class="loading">Loading explanation...</div>';

    try {
        const url = `${API_BASE}/explain/${transactionId}`;
        console.log('Fetching:', url);
        const response = await fetch(url);
        const data = await response.json();

        console.log('SHAP data received:', data);

        if (data.error) {
            content.innerHTML = `<div class="loading">Error: ${data.error}</div>`;
            return;
        }

        // Render SHAP explanation
        const maxContribution = Math.max(...data.contributions.map(c => Math.abs(c.contribution)));
        console.log('Max contribution:', maxContribution);

        content.innerHTML = `
            <div class="shap-summary">
                <h3>Transaction #${data.transaction_id}</h3>
                <p><strong>Fraud Score:</strong> ${(data.prediction * 100).toFixed(2)}%</p>
                <p>${data.explanation}</p>
            </div>
            
            <h3>Top Contributing Features</h3>
            <div class="shap-explanation">
                ${data.contributions.slice(0, 10).map(feature => {
            const percentage = (Math.abs(feature.contribution) / maxContribution) * 100;
            const isPositive = feature.contribution > 0;

            return `
                        <div class="feature-contribution">
                            <div class="feature-name">${feature.feature}</div>
                            <div class="feature-value">${feature.value.toFixed(4)}</div>
                            <div class="contribution-bar">
                                <div class="contribution-fill ${isPositive ? '' : 'negative'}" 
                                     style="width: ${percentage}%"></div>
                            </div>
                            <div class="feature-value">${isPositive ? '+' : ''}${feature.contribution.toFixed(4)}</div>
                        </div>
                    `;
        }).join('')}
            </div>
            
            <div style="margin-top: 1.5rem; padding: 1rem; background: rgba(139, 92, 246, 0.1); border-radius: 0.5rem;">
                <p style="font-size: 0.875rem; color: var(--text-secondary);">
                    <strong>How to read:</strong> Positive contributions (purple) increase fraud probability. 
                    Negative contributions (red) decrease it. The bar width shows relative importance.
                </p>
            </div>
        `;
        console.log('SHAP explanation rendered successfully');
    } catch (error) {
        console.error('Error loading explanation:', error);
        content.innerHTML = `<div class="loading">Error: ${error.message}</div>`;
    }
}

// Close modal
function closeModal() {
    document.getElementById('shapModal').classList.remove('active');
}

// Initialize modal close handler after DOM loads
document.addEventListener('DOMContentLoaded', () => {
    const modal = document.getElementById('shapModal');
    if (modal) {
        modal.addEventListener('click', (e) => {
            if (e.target.id === 'shapModal') {
                closeModal();
            }
        });
        console.log('Modal event listener initialized');
    } else {
        console.error('Modal element not found!');
    }
});

// Load charts
async function loadCharts() {
    await loadDistributionChart();
    await loadTimelineChart();
}

// Fraud score distribution chart
async function loadDistributionChart() {
    try {
        const response = await fetch(`${API_BASE}/distribution`);
        const data = await response.json();

        const ctx = document.getElementById('distributionChart').getContext('2d');

        if (distributionChart) {
            distributionChart.destroy();
        }

        distributionChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: data.map(d => d.range),
                datasets: [{
                    label: 'Transaction Count',
                    data: data.map(d => d.count),
                    backgroundColor: 'rgba(139, 92, 246, 0.6)',
                    borderColor: 'rgba(139, 92, 246, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: {
                            color: 'rgba(148, 163, 184, 0.1)'
                        },
                        ticks: {
                            color: '#94a3b8'
                        }
                    },
                    x: {
                        grid: {
                            display: false
                        },
                        ticks: {
                            color: '#94a3b8',
                            maxRotation: 45,
                            minRotation: 45
                        }
                    }
                }
            }
        });
    } catch (error) {
        console.error('Error loading distribution chart:', error);
    }
}

// Alerts timeline chart
async function loadTimelineChart() {
    try {
        const response = await fetch(`${API_BASE}/timeline`);
        const data = await response.json();

        const ctx = document.getElementById('timelineChart').getContext('2d');

        if (timelineChart) {
            timelineChart.destroy();
        }

        timelineChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.map(d => `Batch ${d.batch}`),
                datasets: [{
                    label: 'Alert Count',
                    data: data.map(d => d.alerts),
                    borderColor: 'rgba(59, 130, 246, 1)',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: {
                            color: 'rgba(148, 163, 184, 0.1)'
                        },
                        ticks: {
                            color: '#94a3b8'
                        }
                    },
                    x: {
                        grid: {
                            display: false
                        },
                        ticks: {
                            color: '#94a3b8'
                        }
                    }
                }
            }
        });
    } catch (error) {
        console.error('Error loading timeline chart:', error);
    }
}
