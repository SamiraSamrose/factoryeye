/**
 * Chart Visualizations for FactoryEye Dashboard
 * Handles Chart.js initialization and updates
 */

class ChartManager {
    constructor() {
        this.charts = {};
        this.colors = {
            primary: '#1a73e8',
            secondary: '#34a853',
            warning: '#fbbc04',
            danger: '#ea4335',
            info: '#00bcd4'
        };
    }

    /**
     * Initialize time series anomaly chart
     * Shows anomaly detections over time
     */
    initAnomalyTimeSeriesChart(canvasId) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return;

        this.charts.anomalyTimeSeries = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Anomaly Score',
                    data: [],
                    borderColor: this.colors.danger,
                    backgroundColor: 'rgba(234, 67, 53, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1,
                        title: {
                            display: true,
                            text: 'Anomaly Score'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Time'
                        }
                    }
                }
            }
        });
    }

    /**
     * Initialize equipment heatmap chart
     * Shows machine performance as heatmap
     */
    initEquipmentHeatmapChart(canvasId) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return;

        this.charts.equipmentHeatmap = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Temperature',
                    data: [],
                    backgroundColor: this.colors.danger
                }, {
                    label: 'Vibration',
                    data: [],
                    backgroundColor: this.colors.warning
                }, {
                    label: 'Pressure',
                    data: [],
                    backgroundColor: this.colors.info
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Normalized Value'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Machine ID'
                        }
                    }
                }
            }
        });
    }

    /**
     * Initialize RUL prediction chart
     * Shows remaining useful life predictions
     */
    initRULPredictionChart(canvasId) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return;

        this.charts.rulPrediction = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Predicted RUL (hours)',
                    data: [],
                    borderColor: this.colors.primary,
                    backgroundColor: 'rgba(26, 115, 232, 0.1)',
                    tension: 0.4,
                    fill: true
                }, {
                    label: 'Confidence',
                    data: [],
                    borderColor: this.colors.secondary,
                    backgroundColor: 'rgba(52, 168, 83, 0.1)',
                    tension: 0.4,
                    fill: true,
                    yAxisID: 'y1'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    }
                },
                scales: {
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'RUL (hours)'
                        }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        min: 0,
                        max: 1,
                        title: {
                            display: true,
                            text: 'Confidence'
                        },
                        grid: {
                            drawOnChartArea: false
                        }
                    }
                }
            }
        });
    }

    /**
     * Initialize efficiency trend chart
     * Shows efficiency metrics over time
     */
    initEfficiencyTrendChart(canvasId) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return;

        this.charts.efficiencyTrend = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Efficiency Score',
                    data: [],
                    borderColor: this.colors.secondary,
                    backgroundColor: 'rgba(52, 168, 83, 0.1)',
                    tension: 0.4,
                    fill: true
                }, {
                    label: 'Uptime %',
                    data: [],
                    borderColor: this.colors.primary,
                    backgroundColor: 'rgba(26, 115, 232, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Percentage'
                        }
                    }
                }
            }
        });
    }

    /**
     * Update anomaly time series with new data
     * Adds data point to chart
     */
    updateAnomalyTimeSeries(timestamp, score) {
        const chart = this.charts.anomalyTimeSeries;
        if (!chart) return;

        const timeLabel = new Date(timestamp).toLocaleTimeString();
        
        chart.data.labels.push(timeLabel);
        chart.data.datasets[0].data.push(score);

        // Keep only last 50 points
        if (chart.data.labels.length > 50) {
            chart.data.labels.shift();
            chart.data.datasets[0].data.shift();
        }

        chart.update('none');
    }

    /**
     * Load historical data for charts
     * Fetches data from API and populates charts
     */
    async loadHistoricalData(hours = 24) {
        try {
            // Load anomaly data
            const anomalyResponse = await fetch(`/api/v1/anomalies?hours=${hours}`);
            const anomalyData = await anomalyResponse.json();
            this.populateAnomalyChart(anomalyData.anomalies);

            // Load prediction data
            const predictionResponse = await fetch('/api/v1/predictions');
            const predictionData = await predictionResponse.json();
            this.populateRULChart(predictionData.predictions);

        } catch (error) {
            console.error('Error loading historical data:', error);
        }
    }

    /**
     * Populate anomaly chart with historical data
     * Processes and displays anomaly history
     */
    populateAnomalyChart(anomalies) {
        const chart = this.charts.anomalyTimeSeries;
        if (!chart) return;

        chart.data.labels = [];
        chart.data.datasets[0].data = [];

        anomalies.slice(-50).forEach(anomaly => {
            const timeLabel = new Date(anomaly.timestamp).toLocaleTimeString();
            chart.data.labels.push(timeLabel);
            chart.data.datasets[0].data.push(anomaly.anomaly_score);
        });

        chart.update();
    }

    /**
     * Populate RUL chart with prediction data
     * Displays RUL forecasts for machines
     */
    populateRULChart(predictions) {
        const chart = this.charts.rulPrediction;
        if (!chart) return;

        chart.data.labels = [];
        chart.data.datasets[0].data = [];
        chart.data.datasets[1].data = [];

        predictions.slice(0, 20).forEach(pred => {
            chart.data.labels.push(pred.machine_id);
            chart.data.datasets[0].data.push(pred.rul_hours);
            chart.data.datasets[1].data.push(pred.confidence);
        });

        chart.update();
    }

    /**
     * Initialize all charts
     * Called on page load
     */
    initializeAll() {
        this.initAnomalyTimeSeriesChart('anomalyTimeSeriesChart');
        this.initEquipmentHeatmapChart('equipmentHeatmapChart');
        this.initRULPredictionChart('rulPredictionChart');
        this.initEfficiencyTrendChart('efficiencyTrendChart');
        
        // Load historical data
        this.loadHistoricalData();
    }
}

// Make ChartManager available globally
window.chartManager = new ChartManager();