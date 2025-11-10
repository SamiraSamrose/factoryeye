/**
 * FactoryEye Dashboard JavaScript
 * Handles real-time data updates and user interactions
 */

class FactoryEyeDashboard {
    constructor() {
        this.websocket = null;
        this.reconnectInterval = 5000;
        this.charts = {};
        this.init();
    }

    /**
     * Initialize dashboard
     * Sets up WebSocket connection and loads initial data
     */
    async init() {
        console.log('Initializing FactoryEye Dashboard...');
        
        // Connect to WebSocket
        this.connectWebSocket();
        
        // Load initial data
        await this.loadDashboardStats();
        await this.loadRecentAnomalies();
        
        // Initialize charts
        this.initializeCharts();
        
        // Set up periodic refresh
        setInterval(() => this.loadDashboardStats(), 30000);
        setInterval(() => this.loadRecentAnomalies(), 60000);
    }

    /**
     * Connect to WebSocket for real-time updates
     * Handles connection, reconnection, and message processing
     */
    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        console.log(`Connecting to WebSocket: ${wsUrl}`);
        
        this.websocket = new WebSocket(wsUrl);
        
        this.websocket.onopen = () => {
            console.log('WebSocket connected');
            this.updateConnectionStatus(true);
        };
        
        this.websocket.onmessage = (event) => {
            const message = JSON.parse(event.data);
            this.handleWebSocketMessage(message);
        };
        
        this.websocket.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.updateConnectionStatus(false);
        };
        
        this.websocket.onclose = () => {
            console.log('WebSocket disconnected');
            this.updateConnectionStatus(false);
            
            // Attempt reconnection
            setTimeout(() => this.connectWebSocket(), this.reconnectInterval);
        };
    }

    /**
     * Handle incoming WebSocket messages
     * Processes anomalies and updates UI
     */
    handleWebSocketMessage(message) {
        console.log('WebSocket message:', message);
        
        if (message.type === 'anomaly') {
            this.handleAnomalyUpdate(message.data);
        } else if (message.type === 'metric') {
            this.handleMetricUpdate(message.data);
        }
    }

    /**
     * Handle anomaly updates from WebSocket
     * Adds new anomaly to list and shows notification
     */
    handleAnomalyUpdate(anomalyData) {
        // Add to anomaly list
        this.addAnomalyToList(anomalyData);
        
        // Show notification for critical anomalies
        if (anomalyData.anomaly_score >= 0.9) {
            this.showNotification(
                'Critical Anomaly Detected',
                `Sensor ${anomalyData.sensor_id}: Score ${anomalyData.anomaly_score.toFixed(2)}`,
                'danger'
            );
        }
        
        // Refresh dashboard stats
        this.loadDashboardStats();
    }

    /**
     * Load dashboard statistics from API
     * Updates stat cards with latest data
     */
    async loadDashboardStats() {
        try {
            const response = await fetch('/api/v1/dashboard/stats');
            const data = await response.json();
            
            this.updateStatCards(data.stats);
        } catch (error) {
            console.error('Error loading dashboard stats:', error);
        }
    }

    /**
     * Update stat cards with new data
     * Animates value changes
     */
    updateStatCards(stats) {
        this.updateStatCard('total-sensors', stats.total_sensors || 0);
        this.updateStatCard('total-machines', stats.total_machines || 0);
        this.updateStatCard('total-anomalies', stats.total_anomalies || 0);
        this.updateStatCard('urgent-maintenance', stats.urgent_maintenance_count || 0);
    }

    /**
     * Update individual stat card
     * Handles animation and formatting
     */
    updateStatCard(elementId, value) {
        const element = document.getElementById(elementId);
        if (element) {
            const currentValue = parseInt(element.textContent) || 0;
            this.animateValue(element, currentValue, value, 500);
        }
    }

    /**
     * Animate value change in element
     * Creates smooth transition effect
     */
    animateValue(element, start, end, duration) {
        const startTime = performance.now();
        
        const updateValue = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            const current = Math.floor(start + (end - start) * progress);
            element.textContent = current;
            
            if (progress < 1) {
                requestAnimationFrame(updateValue);
            }
        };
        
        requestAnimationFrame(updateValue);
    }

    /**
     * Load recent anomalies from API
     * Populates anomaly list
     */
    async loadRecentAnomalies() {
        try {
            const response = await fetch('/api/v1/anomalies?hours=24');
            const data = await response.json();
            
            this.displayAnomalies(data.anomalies);
        } catch (error) {
            console.error('Error loading anomalies:', error);
        }
    }

    /**
     * Display anomalies in list
     * Clears and repopulates anomaly container
     */
    displayAnomalies(anomalies) {
        const container = document.getElementById('anomaly-list');
        if (!container) return;
        
        container.innerHTML = '';
        
        if (anomalies.length === 0) {
            container.innerHTML = '<div class="no-data">No anomalies detected in the last 24 hours</div>';
            return;
        }
        
        anomalies.slice(0, 10).forEach(anomaly => {
            container.appendChild(this.createAnomalyElement(anomaly));
        });
    }

    /**
     * Create anomaly list item element
     * Returns DOM element for anomaly
     */
    createAnomalyElement(anomaly) {
        const item = document.createElement('div');
        item.className = 'anomaly-item';
        
        const severity = this.determineSeverity(anomaly.anomaly_score);
        const timestamp = new Date(anomaly.timestamp).toLocaleString();
        
        item.innerHTML = `
            <div class="anomaly-info">
                <div class="anomaly-sensor">Sensor: ${anomaly.sensor_id}</div>
                <div class="anomaly-time">${timestamp}</div>
            </div>
            <span class="severity-badge severity-${severity}">
                ${severity} (${anomaly.anomaly_score.toFixed(2)})
            </span>
        `;
        
        return item;
    }

    /**
     * Add single anomaly to list
     * Prepends new anomaly to existing list
     */
    addAnomalyToList(anomaly) {
        const container = document.getElementById('anomaly-list');
        if (!container) return;
        
        const element = this.createAnomalyElement(anomaly);
        container.insertBefore(element, container.firstChild);
        
        // Remove last item if list is too long
        if (container.children.length > 10) {
            container.removeChild(container.lastChild);
        }
    }

    /**
     * Determine severity level from score
     * Maps score to severity category
     */
    determineSeverity(score) {
        if (score >= 0.95) return 'critical';
        if (score >= 0.90) return 'high';
        if (score >= 0.85) return 'medium';
        return 'low';
    }

    /**
     * Initialize chart visualizations
     * Sets up Chart.js instances
     */
    initializeCharts() {
        // This will be implemented in charts.js
        console.log('Initializing charts...');
    }

    /**
     * Update connection status indicator
     * Shows visual feedback for WebSocket state
     */
    updateConnectionStatus(connected) {
        const statusDot = document.querySelector('.status-dot');
        const statusText = document.querySelector('.status-text');
        
        if (statusDot) {
            statusDot.style.backgroundColor = connected ? 
                'var(--secondary-color)' : 'var(--danger-color)';
        }
        
        if (statusText) {
            statusText.textContent = connected ? 'Connected' : 'Disconnected';
        }
    }

    /**
     * Show notification to user
     * Displays alert banner with message
     */
    showNotification(title, message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <strong>${title}</strong><br>
            ${message}
        `;
        
        document.body.appendChild(notification);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            notification.remove();
        }, 5000);
    }
}

// Initialize dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new FactoryEyeDashboard();
});