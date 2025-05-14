"""
HTML template module for generating the dashboard components.
"""

# Base HTML template for the dashboard (JavaScript parts)
JAVASCRIPT_TEMPLATE = """
        // Weekly Activity Chart
        function initWeeklyActivityChart(data) {
            const ctx = document.getElementById('weeklyActivityChart').getContext('2d');
            const chart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: data.map(d => d.WeekLabel),
                    datasets: [{
                        label: 'Sessions per Week',
                        data: data.map(d => d.Sessions),
                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        borderColor: 'rgba(75, 192, 192, 1)',
                        borderWidth: 2,
                        tension: 0.1,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Number of Sessions'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Week'
                            }
                        }
                    }
                }
            });
        }

        // Mouse Analysis
        function initMouseAnalysis(progressionData, sessionData) {
            // Set up mouse selector
            const mouseSelector = document.getElementById('mouseSelector');
            
            if (!mouseSelector) return;
            
            // Get unique mice
            const mice = [...new Set(progressionData.map(d => d.Mouse))].sort();
            
            // Populate selector
            mouseSelector.innerHTML = mice.map(mouse => 
                `<option value="${mouse}">${mouse}</option>`
            ).join('');
            
            // Initial mouse
            const initialMouse = mice[0];
            
            // Function to update charts for selected mouse
            const updateMouseCharts = (mouseId) => {
                // Filter data for the selected mouse
                const mouseProgression = progressionData.filter(d => d.Mouse === mouseId);
                const mouseSessions = sessionData.filter(d => d.Mouse === mouseId);
                
                // Update progression chart
                updateMouseProgressionChart(mouseProgression);
                
                // Update mouse calendar
                updateMouseCalendar(mouseSessions);
                
                // Update sessions table
                updateMouseSessionsTable(mouseSessions);
            };
            
            // Event listener for mouse selection
            mouseSelector.addEventListener('change', (e) => {
                updateMouseCharts(e.target.value);
            });
            
            // Initialize with first mouse
            updateMouseCharts(initialMouse);
        }

        // Update Mouse Progression Chart
        function updateMouseProgressionChart(mouseData) {
            // Sort data by date
            mouseData.sort((a, b) => new Date(a.Date) - new Date(b.Date));
            
            // Get unique phases
            const phases = [...new Set(mouseData.map(d => d.Phase))].filter(p => p !== null);
            
            // Create chart data
            const chartData = {
                labels: mouseData.map(d => d.Date),
                datasets: phases.map(phase => {
                    // Get color for phase
                    const phaseColor = getPhaseColor(phase);
                    
                    // Create dataset
                    return {
                        label: `Phase ${phase}`,
                        data: mouseData.map(d => d.Phase === phase ? phase : null),
                        borderColor: phaseColor,
                        backgroundColor: phaseColor.replace('1)', '0.2)'),
                        pointRadius: 6,
                        pointHoverRadius: 8,
                        showLine: false
                    };
                })
            };
            
            // Clear existing chart
            if (window.mouseProgressionChart) {
                window.mouseProgressionChart.destroy();
            }
            
            // Create new chart
            const ctx = document.getElementById('mouseProgressionChart').getContext('2d');
            window.mouseProgressionChart = new Chart(ctx, {
                type: 'scatter',
                data: chartData,
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            type: 'time',
                            time: {
                                unit: 'day',
                                tooltipFormat: 'MMM d, yyyy',
                                displayFormats: {
                                    day: 'MMM d'
                                }
                            },
                            title: {
                                display: true,
                                text: 'Date'
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Phase'
                            },
                            min: 0,
                            max: Math.max(...phases.map(p => parseInt(p))) + 1,
                            ticks: {
                                stepSize: 1
                            }
                        }
                    },
                    plugins: {
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    const point = context.raw;
                                    return `Phase: ${point}`;
                                }
                            }
                        },
                        legend: {
                            display: true,
                            position: 'bottom'
                        }
                    }
                }
            });
        }

        // Update Mouse Calendar
        function updateMouseCalendar(mouseData) {
            // Group sessions by date
            const sessionsByDate = {};
            mouseData.forEach(session => {
                if (!sessionsByDate[session.Date]) {
                    sessionsByDate[session.Date] = [];
                }
                sessionsByDate[session.Date].push(session);
            });
            
            // Get date range
            const dates = Object.keys(sessionsByDate).sort();
            if (dates.length === 0) {
                // No data
                document.getElementById('mouseCalendarContainer').innerHTML = 
                    '<div class="alert alert-info">No sessions found for this mouse.</div>';
                return;
            }
            
            // Create calendar
            let calendarHtml = '<div class="table-responsive"><table class="table table-bordered">';
            
            // Add header
            calendarHtml += '<thead><tr><th>Date</th><th>Sessions</th><th>Phases</th></tr></thead>';
            
            // Add rows
            calendarHtml += '<tbody>';
            dates.forEach(date => {
                const sessions = sessionsByDate[date];
                const phases = [...new Set(sessions.map(s => s.Phase))].filter(p => p !== null);
                
                calendarHtml += `<tr>
                    <td>${formatDate(date)}</td>
                    <td>${sessions.length}</td>
                    <td>${phases.map(p => `<span class="badge phase-${p}">Phase ${p}</span>`).join(' ')}</td>
                </tr>`;
            });
            
            calendarHtml += '</tbody></table></div>';
            
            // Update container
            document.getElementById('mouseCalendarContainer').innerHTML = calendarHtml;
        }

        // Update Mouse Sessions Table
        function updateMouseSessionsTable(mouseData) {
            // Sort by date and time
            mouseData.sort((a, b) => {
                if (a.Date === b.Date) {
                    return a.Time.localeCompare(b.Time);
                }
                return new Date(b.Date) - new Date(a.Date); // Most recent first
            });
            
            // Create rows
            const tableBody = document.getElementById('mouseSessionsTableBody');
            tableBody.innerHTML = '';
            
            mouseData.forEach(session => {
                const row = tableBody.insertRow();
                
                // Date
                const dateCell = row.insertCell();
                dateCell.textContent = formatDate(session.Date);
                
                // Time
                const timeCell = row.insertCell();
                timeCell.textContent = formatTime(session.Time);
                
                // Session ID
                const sessionIdCell = row.insertCell();
                sessionIdCell.textContent = session['Session ID'];
                
                // Phase
                const phaseCell = row.insertCell();
                const phase = session.Phase;
                phaseCell.innerHTML = phase ? 
                    `<span class="badge phase-${phase}">Phase ${phase}</span>` : 
                    '<span class="badge bg-secondary">Unknown</span>';
                
                // Trials
                const trialsCell = row.insertCell();
                trialsCell.textContent = session['Total Trials'] || 'N/A';
                
                // Video Length
                const videoLengthCell = row.insertCell();
                videoLengthCell.textContent = session['Video Length (min)'] ? 
                    `${session['Video Length (min)']} min` : 'N/A';
            });
        }

        // Phase Analysis
        function initPhaseAnalysis(progressionData, sessionData) {
            // Set up phase selector
            const phaseSelector = document.getElementById('phaseSelector');
            
            if (!phaseSelector) return;
            
            // Get unique phases
            const phases = [...new Set(progressionData.map(d => d.Phase))]
                .filter(p => p !== null && p !== undefined)
                .sort((a, b) => {
                    // Try to sort numerically if possible
                    const numA = parseInt(a);
                    const numB = parseInt(b);
                    if (!isNaN(numA) && !isNaN(numB)) {
                        return numA - numB;
                    }
                    return String(a).localeCompare(String(b));
                });
            
            // Populate selector
            phaseSelector.innerHTML = phases.map(phase => 
                `<option value="${phase}">Phase ${phase}</option>`
            ).join('');
            
            // Initial phase
            const initialPhase = phases[0];
            
            // Function to update charts for selected phase
            const updatePhaseCharts = (phase) => {
                // Filter data for the selected phase
                const phaseProgression = progressionData.filter(d => d.Phase == phase);
                const phaseSessions = sessionData.filter(d => d.Phase == phase);
                
                // Update mice distribution chart
                updatePhaseDistributionByMiceChart(phaseProgression);
                
                // Update phase timeline chart
                updatePhaseTimelineChart(phaseProgression);
                
                // Update sessions table
                updatePhaseSessionsTable(phaseSessions);
            };
            
            // Event listener for phase selection
            phaseSelector.addEventListener('change', (e) => {
                updatePhaseCharts(e.target.value);
            });
            
            // Initialize with first phase
            updatePhaseCharts(initialPhase);
        }

        // Update Phase Distribution By Mice Chart
        function updatePhaseDistributionByMiceChart(phaseData) {
            // Count sessions per mouse
            const miceCounts = {};
            phaseData.forEach(session => {
                if (!miceCounts[session.Mouse]) {
                    miceCounts[session.Mouse] = 0;
                }
                miceCounts[session.Mouse]++;
            });
            
            // Sort mice by count
            const sortedMice = Object.keys(miceCounts).sort((a, b) => miceCounts[b] - miceCounts[a]);
            
            // Create dataset
            const chartData = {
                labels: sortedMice,
                datasets: [{
                    label: 'Sessions',
                    data: sortedMice.map(mouse => miceCounts[mouse]),
                    backgroundColor: 'rgba(75, 192, 192, 0.7)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1
                }]
            };
            
            // Clear existing chart
            if (window.phaseDistributionByMiceChart) {
                window.phaseDistributionByMiceChart.destroy();
            }
            
            // Create new chart
            const ctx = document.getElementById('phaseDistributionByMiceChart').getContext('2d');
            window.phaseDistributionByMiceChart = new Chart(ctx, {
                type: 'bar',
                data: chartData,
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Number of Sessions'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Mouse'
                            }
                        }
                    }
                }
            });
        }

        // Update Phase Timeline Chart
        function updatePhaseTimelineChart(phaseData) {
            // Group sessions by date
            const sessionsByDate = {};
            phaseData.forEach(session => {
                if (!sessionsByDate[session.Date]) {
                    sessionsByDate[session.Date] = 0;
                }
                sessionsByDate[session.Date]++;
            });
            
            // Sort dates
            const sortedDates = Object.keys(sessionsByDate).sort();
            
            // Create dataset
            const chartData = {
                labels: sortedDates,
                datasets: [{
                    label: 'Sessions',
                    data: sortedDates.map(date => sessionsByDate[date]),
                    backgroundColor: 'rgba(54, 162, 235, 0.2)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 2,
                    tension: 0.1,
                    fill: true
                }]
            };
            
            // Clear existing chart
            if (window.phaseTimelineChart) {
                window.phaseTimelineChart.destroy();
            }
            
            // Create new chart
            const ctx = document.getElementById('phaseTimelineChart').getContext('2d');
            window.phaseTimelineChart = new Chart(ctx, {
                type: 'line',
                data: chartData,
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            type: 'time',
                            time: {
                                unit: 'day',
                                tooltipFormat: 'MMM d, yyyy',
                                displayFormats: {
                                    day: 'MMM d'
                                }
                            },
                            title: {
                                display: true,
                                text: 'Date'
                            }
                        },
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Number of Sessions'
                            }
                        }
                    }
                }
            });
        }

        // Update Phase Sessions Table
        function updatePhaseSessionsTable(phaseData) {
            // Sort by date and time
            phaseData.sort((a, b) => {
                if (a.Date === b.Date) {
                    return a.Time.localeCompare(b.Time);
                }
                return new Date(b.Date) - new Date(a.Date); // Most recent first
            });
            
            // Create rows
            const tableBody = document.getElementById('phaseSessionsTableBody');
            tableBody.innerHTML = '';
            
            phaseData.forEach(session => {
                const row = tableBody.insertRow();
                
                // Date
                const dateCell = row.insertCell();
                dateCell.textContent = formatDate(session.Date);
                
                // Time
                const timeCell = row.insertCell();
                timeCell.textContent = formatTime(session.Time);
                
                // Mouse
                const mouseCell = row.insertCell();
                mouseCell.textContent = session.Mouse;
                
                // Session ID
                const sessionIdCell = row.insertCell();
                sessionIdCell.textContent = session['Session ID'];
                
                // Trials
                const trialsCell = row.insertCell();
                trialsCell.textContent = session['Total Trials'] || 'N/A';
                
                // Video Length
                const videoLengthCell = row.insertCell();
                videoLengthCell.textContent = session['Video Length (min)'] ? 
                    `${session['Video Length (min)']} min` : 'N/A';
            });
        }

        // Session Explorer
        function initSessionExplorer(sessionData) {
            // Initialize DataTable
            const table = $('#allSessionsTable').DataTable({
                data: sessionData,
                columns: [
                    { data: 'Date' },
                    { data: 'Time' },
                    { data: 'Mouse' },
                    { data: 'Session ID' },
                    { 
                        data: 'Phase',
                        render: function(data) {
                            return data ? 
                                `<span class="badge phase-${data}">Phase ${data}</span>` : 
                                '<span class="badge bg-secondary">Unknown</span>';
                        }
                    },
                    { data: 'Total Trials' },
                    { data: 'Video Length (min)' },
                    { 
                        data: null,
                        render: function(data) {
                            return '<button class="btn btn-sm btn-info view-details" data-id="' + data['Session ID'] + '">View</button>';
                        }
                    }
                ],
                order: [[0, 'desc'], [1, 'desc']],
                pageLength: 25,
                responsive: true
            });
            
            // Session details modal
            $('#allSessionsTable').on('click', '.view-details', function() {
                const sessionId = $(this).data('id');
                const session = sessionData.find(s => s['Session ID'] === sessionId);
                
                if (session) {
                    $('#sessionModalLabel').text(`Session Details: ${sessionId}`);
                    
                    // Build modal content
                    let content = `
                        <div class="row">
                            <div class="col-md-6">
                                <p><strong>Date:</strong> ${formatDate(session.Date)}</p>
                                <p><strong>Time:</strong> ${formatTime(session.Time)}</p>
                                <p><strong>Mouse:</strong> ${session.Mouse}</p>
                                <p><strong>Phase:</strong> <span class="badge phase-${session.Phase || 'unknown'}">${session.Phase || 'Unknown'}</span></p>
                            </div>
                            <div class="col-md-6">
                                <p><strong>Total Trials:</strong> ${session['Total Trials'] || 'N/A'}</p>
                                <p><strong>Video Length:</strong> ${session['Video Length (min)'] || 'N/A'} minutes</p>
                                <p><strong>Directory:</strong> ${session.directory || 'N/A'}</p>
                            </div>
                        </div>
                        <hr>
                        <div class="row">
                            <div class="col-12">
                                <h6>Additional Information</h6>
                                <p><strong>Session ID:</strong> ${session['Session ID']}</p>
                                <p><strong>NWB File:</strong> ${session['NWB_file'] || 'N/A'}</p>
                                <p><strong>Is Complete:</strong> ${session['Is Complete'] ? 'Yes' : 'No'}</p>
                            </div>
                        </div>
                    `;
                    
                    $('#sessionModalBody').html(content);
                    const modal = new bootstrap.Modal(document.getElementById('sessionDetailsModal'));
                    modal.show();
                }
            });
            
            // Filter functionality
            const applyFilters = () => {
                const startDate = $('#startDateFilter').val();
                const endDate = $('#endDateFilter').val();
                const selectedMice = Array.from($('#mouseFilter option:selected'))
                    .map(opt => opt.value);
                const selectedPhases = Array.from($('#phaseFilter option:selected'))
                    .map(opt => opt.value);
                
                // Custom filtering function
                $.fn.dataTable.ext.search.push(
                    function(settings, data, dataIndex, row) {
                        // Date filter
                        if (startDate && endDate) {
                            const date = new Date(row.Date);
                            const min = new Date(startDate);
                            const max = new Date(endDate);
                            if (date < min || date > max) {
                                return false;
                            }
                        }
                        
                        // Mouse filter
                        if (!selectedMice.includes('all') && !selectedMice.includes(row.Mouse)) {
                            return false;
                        }
                        
                        // Phase filter
                        if (!selectedPhases.includes('all') && !selectedPhases.includes(row.Phase)) {
                            return false;
                        }
                        
                        return true;
                    }
                );
                
                // Apply filters
                table.draw();
                
                // Remove the custom filtering function
                $.fn.dataTable.ext.search.pop();
            };
            
            // Event listeners for filter buttons
            $('#applyFilters').on('click', applyFilters);
            
            $('#resetFilters').on('click', function() {
                $('#startDateFilter').val('');
                $('#endDateFilter').val('');
                $('#mouseFilter').val(['all']);
                $('#phaseFilter').val(['all']);
                table.search('').columns().search('').draw();
            });
        }

        // Utility Functions
        function formatDate(dateStr) {
            const date = new Date(dateStr);
            return date.toLocaleDateString('en-GB', { 
                day: '2-digit', 
                month: 'short', 
                year: 'numeric' 
            });
        }
        
        function formatTime(timeStr) {
            // Format HHMMSS to HH:MM:SS
            if (timeStr && timeStr.length === 6) {
                return `${timeStr.slice(0, 2)}:${timeStr.slice(2, 4)}:${timeStr.slice(4, 6)}`;
            }
            return timeStr;
        }
        
        function getPhaseColor(phase) {
            const phaseColors = {
                '1': 'rgba(0, 123, 255, 1)',
                '2': 'rgba(40, 167, 69, 1)',
                '3': 'rgba(23, 162, 184, 1)',
                '4': 'rgba(255, 193, 7, 1)',
                '5': 'rgba(220, 53, 69, 1)',
                '6': 'rgba(102, 16, 242, 1)',
                '7': 'rgba(253, 126, 20, 1)',
                '8': 'rgba(32, 201, 151, 1)',
                '9': 'rgba(232, 62, 140, 1)',
                '10': 'rgba(111, 66, 193, 1)',
                '3b': 'rgba(19, 132, 150, 1)',
                '4b': 'rgba(211, 158, 0, 1)',
                'test': 'rgba(108, 117, 125, 1)'
            };
            
            return phaseColors[phase] || 'rgba(108, 117, 125, 1)';
        }
        
        // Navigation setup
        function setupNavigation() {
            // Make navbar links smooth scroll
            document.querySelectorAll('.navbar-nav a.nav-link').forEach(link => {
                link.addEventListener('click', function(e) {
                    e.preventDefault();
                    const targetId = this.getAttribute('href');
                    const targetElement = document.querySelector(targetId);
                    
                    if (targetElement) {
                        window.scrollTo({
                            top: targetElement.offsetTop - 60,
                            behavior: 'smooth'
                        });
                    }
                });
            });
        }
    </script>
</body>
</html>
"""

def generate_html_template(cohort_meta, date_generated, cohort_directory):
    """
    Generates the full HTML template with the appropriate placeholders filled in.
    
    Args:
        cohort_meta: Dictionary containing cohort metadata
        date_generated: Date string when the dashboard was generated
        cohort_directory: Path to the cohort directory
        
    Returns:
        Complete HTML template as a string
    """
    from datetime import datetime
    
    # Extract metadata
    cohort_name = cohort_meta.get("cohort_name", "Unknown Cohort")
    total_mice = cohort_meta.get("total_mice", 0)
    total_sessions = cohort_meta.get("total_sessions", 0)
    
    # Format date range
    date_range = cohort_meta.get("date_range", [None, None])
    date_range_start = date_range[0] if date_range[0] else "N/A"
    date_range_end = date_range[1] if date_range[1] else "N/A"
    
    # Generate mouse options for dropdowns
    mice_list = cohort_meta.get("mice_list", [])
    mouse_options = "\n".join([f'<option value="{mouse}">{mouse}</option>' for mouse in mice_list])
    mouse_options_single = "\n".join([f'<option value="{mouse}">{mouse}</option>' for mouse in mice_list])
    
    # Generate phase options
    # This should be dynamic based on actual phases in your data
    phase_list = ["1", "2", "3", "3b", "4", "4b", "5", "6", "7", "8", "9", "10", "test"]
    phase_options = "\n".join([f'<option value="{phase}">Phase {phase}</option>' for phase in phase_list])
    phase_options_all = "\n".join([f'<option value="{phase}">Phase {phase}</option>' for phase in phase_list])
    
    # Generate month options
    # This should be dynamic based on your date range
    month_options = ""
    if date_range[0] and date_range[1]:
        try:
            start_date = datetime.strptime(date_range[0], "%Y-%m-%d")
            end_date = datetime.strptime(date_range[1], "%Y-%m-%d")
            
            current_month = datetime(start_date.year, start_date.month, 1)
            while current_month <= end_date:
                month_name = current_month.strftime("%B %Y")
                month_value = current_month.strftime("%Y-%m")
                month_options += f'<option value="{month_value}">{month_name}</option>\n'
                
                # Move to next month
                if current_month.month == 12:
                    current_month = datetime(current_month.year + 1, 1, 1)
                else:
                    current_month = datetime(current_month.year, current_month.month + 1, 1)
        except:
            # Fallback if date parsing fails
            month_options = "<option value='all'>All Months</option>"
    
    # Default date range for filters
    default_start_date = date_range[0] if date_range[0] else ""
    default_end_date = date_range[1] if date_range[1] else ""
    
    # Get the base HTML template
    with open('cohort_visualizer/visualization/html_templates/base_template.html', 'r') as f:
        base_template = f.read()
    
    # Combine with JavaScript
    full_template = base_template + JAVASCRIPT_TEMPLATE
    
    # Replace placeholders
    template = full_template.format(
        cohort_name=cohort_name,
        date_generated=date_generated,
        total_mice=total_mice,
        total_sessions=total_sessions,
        date_range_start=date_range_start,
        date_range_end=date_range_end,
        mouse_options=mouse_options,
        mouse_options_single=mouse_options_single,
        phase_options=phase_options,
        phase_options_all=phase_options_all,
        month_options=month_options,
        default_start_date=default_start_date,
        default_end_date=default_end_date,
        cohort_directory=str(cohort_directory)
    )
    
    return template