// AuraCode Frontend - Style Implementation
// Configuration and globals
const CONFIG = {
    API_BASE: window.location.origin,
    POLLING_INTERVAL: 5000,
    MAX_MESSAGE_LENGTH: 2000,
    TYPING_DELAY: 1000,
};

// State management
const state = {
    currentUser: null,
    currentAgent: null,
    currentView: 'discover',
    agents: [],
    messages: [],
    sessionModelOverride: null,
    sessionId: null,
    isTyping: false,
    currentPrecisionMode: false,
    authMode: 'login',
    creationMode: 'scratch',
    selectedSystemType: 'ceaf',
    selectedPrebuiltAgent: null,
    isAuthenticated: false,
    sidebarCollapsed: false,
    currentEditingAgentId: null,
    selectedMemoryFile: null,
};

// DOM elements cache
const elements = {};

// Initialize application
document.addEventListener('DOMContentLoaded', async () => {
    console.log('🚀 Aura initialized');

    debugger;

    // Cache DOM elements
    cacheElements();

    // Initialize marketplace
    initializeMarketplace();

    // Setup event listeners
    setupEventListeners();
    
    // Check authentication
    await checkAuth();
    
    // Initialize UI
    initializeUI();
    
    // Load initial data
    await loadInitialData();

});

// Cache frequently used DOM elements
function cacheElements() {
    elements.sidebar = document.getElementById('sidebar');
    elements.sidebarToggle = document.getElementById('sidebar-toggle');
    elements.mainContent = document.getElementById('main-content');
    elements.userProfile = document.getElementById('user-profile');
    elements.userMenu = document.getElementById('user-menu');
    elements.usernameDisplay = document.getElementById('username-display');
    elements.userInitial = document.getElementById('user-initial');
    elements.agentsList = document.getElementById('agents-list');
    elements.precisionModeIndicator = document.getElementById('precision-mode-indicator');
    elements.agentCount = document.getElementById('agent-count');
   const myAgentsTabElement = document.getElementById('my-agents-tab');
    if (myAgentsTabElement) {
        myAgentsTabElement.addEventListener('click', (event) => {
            event.preventDefault(); // Prevents the '#' from being added to the URL
            console.log('CLICK DETECTED on #my-agents-tab!', event);
            switchView('my-agents');
        });
    } else {
        // This error is still useful for debugging if the element is truly missing from the HTML.
        console.error("FATAL: Could not find 'my-agents-tab' to attach a click listener.");
    }

    elements.myAgentsView = document.getElementById('my-agents-view');
    elements.myAgentsGrid = document.getElementById('my-agents-grid');
    elements.setPriceModal = document.getElementById('set-price-modal');
    elements.priceAgentName = document.getElementById('price-agent-name');
    elements.agentPriceInput = document.getElementById('agent-price-input');

    elements.menuBtn = document.getElementById('menu-btn');
    elements.agentDropdownMenu = document.getElementById('agent-dropdown-menu');

    // Profile Modal
    elements.profileModal = document.getElementById('profile-modal');
    elements.profileModalAvatarImg = document.getElementById('profile-modal-avatar-img');
    elements.profileModalAvatarInitial = document.getElementById('profile-modal-avatar-initial');
    elements.profileModalAgentName = document.getElementById('profile-modal-agent-name');
    elements.avatarUploadInput = document.getElementById('avatar-upload-input');

    // Files Modal
    elements.filesModal = document.getElementById('files-modal');
    elements.filesListContainer = document.getElementById('files-list-container');
    elements.knowledgeFileInput = document.getElementById('knowledge-file-input');
    elements.knowledgeFileUploadArea = document.getElementById('knowledge-file-upload-area');

    // Model Selector
    elements.chatModelSelector = document.getElementById('chat-model-selector');

    // Views
    elements.discoverView = document.getElementById('discover-view');
    elements.chatView = document.getElementById('chat-view');
    elements.createView = document.getElementById('create-view');
    elements.myAgentsView = document.getElementById('my-agents-view');

    // Nav items
    elements.discoverTab = document.getElementById('discover-tab');
    elements.chatsTab = document.getElementById('chats-tab');
    elements.createAgentBtn = document.getElementById('create-agent-btn');

    // Novo toggle
    elements.agentPublishMenuBtn = document.getElementById('agent-publish-menu-btn');
    elements.publishModal = document.getElementById('publish-modal');
    elements.publishAgentName = document.getElementById('publish-agent-name');
    elements.publishChangelog = document.getElementById('publish-changelog');
    elements.confirmPublishBtn = document.getElementById('confirm-publish-btn');
    elements.publishIncludeHistory = document.getElementById('publish-include-history');

    // Chat elements
    elements.chatMessages = document.getElementById('chat-messages');
    elements.messageInput = document.getElementById('message-input');
    elements.sendBtn = document.getElementById('send-btn');
    elements.currentAgentInfo = document.getElementById('current-agent-info');
    elements.currentAgentName = document.getElementById('current-agent-name');
    elements.currentAgentDescription = document.getElementById('current-agent-description');
    elements.currentAgentAvatar = document.getElementById('current-agent-avatar');
    
    // Auth modal
    elements.authModal = document.getElementById('auth-modal');
    elements.authForm = document.getElementById('auth-form');
    elements.loginTab = document.getElementById('login-tab');
    elements.registerTab = document.getElementById('register-tab');
    elements.authSubmitBtn = document.getElementById('auth-submit-btn');
    elements.errorContainer = document.getElementById('error-container');
    
    // Create form
    elements.agentForm = document.getElementById('agent-form');
    elements.modelSelect = document.getElementById('model-select');
    elements.createButton = document.getElementById('create-button');
    elements.scratchSection = document.getElementById('scratch-section');
    elements.prebuiltSection = document.getElementById('prebuilt-section');
    elements.prebuiltAgentsGrid = document.getElementById('prebuilt-agents-grid');
    
    // Featured content
    elements.featuredGrid = document.getElementById('featured-grid');
    elements.discoverGrid = document.getElementById('discover-grid');
    
    // Loading
    elements.loadingOverlay = document.getElementById('loading-overlay');

       // NEW: Memory Modal Elements
    elements.memoryModal = document.getElementById('memory-modal');

      // ================= START: CACHE NEW ELEMENTS =================
    elements.connectionsDisplay = document.getElementById('connections-display');
    elements.liveMemoryShareToggle = document.getElementById('live-memory-share-toggle');
    elements.liveMemoryNavigateToggle = document.getElementById('live-memory-navigate-toggle');
    // ================== END: CACHE NEW ELEMENTS ==================

        // ================== START: CACHE NEW WIDGET ELEMENTS ==================
    elements.mindWidgetsContainer = document.getElementById('mind-widgets-container');
    elements.memoryWidgetBtn = document.getElementById('memory-widget');
    elements.stateWidgetBtn = document.getElementById('state-widget');
    elements.identityWidgetBtn = document.getElementById('identity-widget');
    elements.memoryPopover = document.getElementById('memory-popover');
    elements.statePopover = document.getElementById('state-popover');
    elements.identityPopover = document.getElementById('identity-popover');
    // =================== END: CACHE NEW WIDGET ELEMENTS ===================
}


function updatePrecisionModeIndicator() {
    if (!elements.precisionModeIndicator) return;

    // Apenas mostre o indicador se o agente for CEAF
    const systemType = getSystemType(state.currentAgent);
    const shouldBeVisible = state.currentPrecisionMode && systemType === 'CEAF';

    elements.precisionModeIndicator.style.display = shouldBeVisible ? 'flex' : 'none';
}

// Setup event listeners
function setupEventListeners() {
    // Sidebar & User Menu
    elements.sidebarToggle?.addEventListener('click', toggleSidebar);
    elements.userProfile?.addEventListener('click', toggleUserMenu);

    // Main Navigation Tabs
    elements.discoverTab?.addEventListener('click', () => switchView('discover'));
    document.getElementById('create-agent-nav-btn')?.addEventListener('click', () => switchView('create'));
    elements.createAgentBtn?.addEventListener('click', () => switchView('create'));

    const myAgentsTabElement = document.getElementById('my-agents-tab');
    if (myAgentsTabElement) {
        myAgentsTabElement.addEventListener('click', (event) => {
            event.preventDefault(); // Prevents the '#' from being added to the URL
            console.log('CLICK DETECTED on #my-agents-tab!', event);
            switchView('my-agents');
        });
    } else {
        // This error is still useful for debugging if the element is truly missing from the HTML.
        console.error("FATAL: Could not find 'my-agents-tab' to attach a click listener.");
    }


    // Chat Header & Menu
    elements.menuBtn?.addEventListener('click', toggleAgentDropdown);
    elements.agentPublishMenuBtn?.addEventListener('click', () => publishAgent(state.currentAgent.agent_id));
    document.getElementById('agent-profile-menu-btn')?.addEventListener('click', showProfileModal);
    document.getElementById('agent-files-menu-btn')?.addEventListener('click', showFilesModal);
    elements.chatModelSelector?.addEventListener('change', updateAgentModel);

    // Chat Input
    elements.messageInput?.addEventListener('keypress', handleMessageKeyPress);
    elements.sendBtn?.addEventListener('click', sendMessage);
    elements.messageInput?.addEventListener('input', handleInputChange);

    // Authentication Modal & Actions
    document.getElementById('login-btn')?.addEventListener('click', showAuthModal);
    document.getElementById('logout-btn')?.addEventListener('click', logout);
    document.getElementById('auth-modal-close')?.addEventListener('click', hideAuthModal);
    elements.authModal?.addEventListener('click', handleModalOverlayClick);
    elements.authForm?.addEventListener('submit', handleAuth);

    // Agent Creation Form
    elements.agentForm?.addEventListener('submit', handleAgentCreation);
    elements.agentForm?.addEventListener('input', updateCreateButton);
    elements.modelSelect?.addEventListener('change', updateCreateButton);

    // Profile & Files Modals
    elements.avatarUploadInput?.addEventListener('change', handleAvatarUpload);
    elements.knowledgeFileInput?.addEventListener('change', handleKnowledgeFileUpload);
    const knowledgeUploadArea = elements.knowledgeFileUploadArea;
    if (knowledgeUploadArea) {
        knowledgeUploadArea.addEventListener('click', () => elements.knowledgeFileInput.click());
        knowledgeUploadArea.addEventListener('dragover', (e) => { e.preventDefault(); e.currentTarget.classList.add('dragover'); });
        knowledgeUploadArea.addEventListener('dragleave', (e) => { e.currentTarget.classList.remove('dragover'); });
        knowledgeUploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            e.currentTarget.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                elements.knowledgeFileInput.files = e.dataTransfer.files;
                handleKnowledgeFileUpload({ target: elements.knowledgeFileInput });
            }
        });
    }

    // Memory Modal
    elements.memoryModal?.addEventListener('click', (e) => {
        if (e.target === elements.memoryModal) closeMemoryModal();
    });
    elements.memoryModal?.querySelector('.tabs')?.addEventListener('click', (e) => {
        if (e.target.matches('.tab')) {
            switchMemoryTab(e.target.dataset.tab);
        }
    });
    document.getElementById('export-btn')?.addEventListener('click', exportMemories);
    document.getElementById('analytics-btn')?.addEventListener('click', loadAnalytics);
    const uploadArea = document.getElementById('file-upload-area');
    if (uploadArea) {
        uploadArea.addEventListener('dragover', (e) => { e.preventDefault(); e.currentTarget.classList.add('dragover'); });
        uploadArea.addEventListener('dragleave', (e) => { e.preventDefault(); e.currentTarget.classList.remove('dragover'); });
        uploadArea.addEventListener('drop', handleFileDrop);
        document.getElementById('memory-file-input').addEventListener('change', handleFileSelect);
    }
    elements.liveMemoryShareToggle?.addEventListener('change', handleLiveMemoryToggle);
    elements.liveMemoryNavigateToggle?.addEventListener('change', handleLiveMemoryToggle);


    // ================= START: NEW WIDGET EVENT LISTENERS =================
    elements.memoryWidgetBtn?.addEventListener('click', (e) => toggleMindPopover(e, 'memory-popover'));
    elements.stateWidgetBtn?.addEventListener('click', (e) => toggleMindPopover(e, 'state-popover'));
    elements.identityWidgetBtn?.addEventListener('click', (e) => toggleMindPopover(e, 'identity-popover'));
    // ================== END: NEW WIDGET EVENT LISTENERS ==================


    // Global Listeners
    document.addEventListener('keydown', handleGlobalKeyDown);
    window.addEventListener('resize', handleResize);
    document.addEventListener('click', handleGlobalClick);
}

// Authentication functions
async function checkAuth() {
    const token = localStorage.getItem('aura_token');
    if (!token) {
        state.isAuthenticated = false;
        updateUIForAuth();
        return;
    }
    
    try {
        const response = await fetch(`${CONFIG.API_BASE}/auth/me`, {
            headers: { 'Authorization': `Bearer ${token}` }
        });
        
        if (response.ok) {
            const user = await response.json();
            state.currentUser = user;
            state.isAuthenticated = true;
            updateUIForAuth();
            console.log('✅ Authentication successful');
        } else {
            localStorage.removeItem('aura_token');
            state.isAuthenticated = false;
            updateUIForAuth();
        }
    } catch (error) {
        console.error('Auth check failed:', error);
        state.isAuthenticated = false;
        updateUIForAuth();
    }
}

function updateUIForAuth() {
    if (state.isAuthenticated && state.currentUser) {
        // Update user display
        if (elements.usernameDisplay) {
            elements.usernameDisplay.textContent = state.currentUser.username;
        }
        if (elements.userInitial) {
            elements.userInitial.textContent = state.currentUser.username[0].toUpperCase();
        }
        
        // Show/hide auth buttons
        document.getElementById('login-btn')?.style.setProperty('display', 'none');
        document.getElementById('logout-btn')?.style.setProperty('display', 'block');
        
        // Enable authenticated features
        elements.createAgentBtn?.removeAttribute('disabled');

        // FIXED: If user just logged in and is on my-agents view, load the agents
        if (state.currentView === 'my-agents') {
            const myAgentsGrid = document.getElementById('my-agents-grid');
            if (myAgentsGrid) {
                // Clear the auth required message and load actual agents
                myAgentsGrid.innerHTML = '<div class="spinner"></div>';
                loadMyAgents();
            }
        }
    } else {
        // Guest mode
        if (elements.usernameDisplay) {
            elements.usernameDisplay.textContent = 'Guest';
        }
        if (elements.userInitial) {
            elements.userInitial.textContent = 'G';
        }
        
        // Show/hide auth buttons
        document.getElementById('login-btn')?.style.setProperty('display', 'block');
        document.getElementById('logout-btn')?.style.setProperty('display', 'none');
        
        // Disable authenticated features
        elements.createAgentBtn?.setAttribute('disabled', 'true');
    }
}

async function handleAuth(event) {
    event.preventDefault();
    
    const formData = new FormData(event.target);
    const username = formData.get('username') || document.getElementById('username').value;
    const password = formData.get('password') || document.getElementById('password').value;
    const email = formData.get('email') || document.getElementById('email').value;
    
    if (!username || !password) {
        showError('Please fill in all required fields');
        return;
    }
    
    showLoading(true);
    clearError();
    
    try {
        const endpoint = state.authMode === 'login' ? '/auth/login' : '/auth/register';
        const body = state.authMode === 'login' 
            ? { username, password }
            : { username, password, email };
        
        const response = await fetch(`${CONFIG.API_BASE}${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        
        const data = await response.json();
        
        if (response.ok) {
            localStorage.setItem('aura_token', data.access_token);
            state.currentUser = data;
            state.isAuthenticated = true;
            updateUIForAuth();
            hideAuthModal();
            
            // Reload recent chats for authenticated user
            await loadRecentChats();
            
            showSuccess(`Welcome${state.authMode === 'register' ? ' to AuraCode' : ' back'}, ${data.username}!`);
            if (state.currentView === 'my-agents') {
                            console.log('User logged in to see My Agents, switching view now.');
                            switchView('my-agents');
            }

        } else {
            showError(data.detail || data.error || 'Authentication failed');
        }
    } catch (error) {
        console.error('Auth error:', error);
        showError('Network error. Please try again.');
    } finally {
        showLoading(false);
    }
}

function logout() {
    localStorage.removeItem('aura_token');
    state.currentUser = null;
    state.isAuthenticated = false;
    state.currentAgent = null;
    state.messages = [];
    state.agents = [];

    updateUIForAuth();
    switchView('discover');

    // Clear agents list
    if (elements.agentsList) {
        elements.agentsList.innerHTML = '';
    }
    if (elements.agentCount) {
        elements.agentCount.textContent = '0';
    }

    showSuccess('Logged out successfully');
}

// Update the loadFeaturedContent function to load actual public agents
async function loadFeaturedContent() {
    if (!elements.featuredGrid) return;

    try {
        showLoading(true);

        // Fetch both featured and public agents at the same time
        const [featuredResponse, publicResponse] = await Promise.all([
            fetch(`${CONFIG.API_BASE}/agents/featured?limit=12`).catch(() => ({ ok: false })),
            fetch(`${CONFIG.API_BASE}/agents/public`).catch(() => ({ ok: false }))
        ]);

        let featuredAgents = [];
        let publicAgents = [];

        if (featuredResponse.ok) {
            featuredAgents = await featuredResponse.json();
        }
        if (publicResponse.ok) {
            publicAgents = await publicResponse.json();
        }

        // Use a Map to merge and de-duplicate agents, giving priority to featured ones.
        const agentMap = new Map();

        // Add public agents first
        publicAgents.forEach(agent => {
            const agentId = agent.agent_id || agent.id;
            agentMap.set(agentId, agent);
        });

        // Add featured agents, overwriting any public ones with the same ID
        featuredAgents.forEach(agent => {
            const agentId = agent.agent_id || agent.id;
            agentMap.set(agentId, agent);
        });

        // Convert the map back to an array
        const combinedAgents = Array.from(agentMap.values());

        // Sort to potentially place featured agents first (optional, but good practice)
        combinedAgents.sort((a, b) => {
            const aIsFeatured = featuredAgents.some(f => (f.agent_id || f.id) === (a.agent_id || a.id));
            const bIsFeatured = featuredAgents.some(f => (f.agent_id || f.id) === (b.agent_id || b.id));
            if (aIsFeatured && !bIsFeatured) return -1;
            if (!aIsFeatured && bIsFeatured) return 1;
            return 0; // Keep original order for same-type agents
        });

        // Render the combined and de-duplicated list
        renderMarketplace(combinedAgents);

    } catch (error) {
        console.error('Error loading marketplace content:', error);
        renderMarketplaceFallback(); // Your existing error handler is good
    } finally {
        showLoading(false);
    }
}

// FIX: This function is now more robust to handle different agent data structures
function renderMarketplace(agents) {
    if (!elements.featuredGrid) return;

    elements.featuredGrid.innerHTML = '';

    if (agents.length === 0) {
        elements.featuredGrid.innerHTML = `
            <div class="no-agents-message" style="grid-column: 1 / -1; text-align: center; padding: 40px;">
                <h3 style="color: var(--text-secondary); margin-bottom: 16px;">No agents available yet</h3>
                <p style="color: var(--text-tertiary);">Be the first to create and share an agent!</p>
            </div>
        `;
        return;
    }

    agents.forEach(agent => {
        const card = document.createElement('div');
        card.className = 'featured-card marketplace-card';

        // FIX: Robustly get agent ID
        const agentId = agent.agent_id || agent.id;
        if (!agentId) return; // Skip rendering if no ID is found

        card.onclick = () => handleAgentSelection(agent);

        const systemType = getSystemType(agent);
        const versionString = agent.version ? ` - v${agent.version}` : '';
        const ownerDisplay = getOwnerDisplay(agent) + versionString;
        const avatarInitial = agent.name ? agent.name[0].toUpperCase() : 'A';
        const avatarUrl = agent.avatar_url ? `${CONFIG.API_BASE}${agent.avatar_url}` : '';
        const cloneCount = agent.clone_count || 0;
        const cloneStatHTML = `
            <div class="stat-item" title="${cloneCount} Clones">
                <svg width="14" height="14" fill="currentColor" viewBox="0 0 16 16"><path d="M4 2a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V2zm2-1a1 1 0 0 0-1 1v8a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1V2a1 1 0 0 0-1-1H6zM2 5a1 1 0 0 0-1 1v8a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1v-1h1v1a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h1v1H2z"/></svg>
                <span>${cloneCount}</span>
            </div>
        `;

        card.innerHTML = `
            <div class="card-header">
                <div class="agent-avatar">
                    <img src="${avatarUrl}" alt="${agent.name} Avatar" style="${!avatarUrl && 'display:none;'}" onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';">
                    <span style="${avatarUrl && 'display:none;'}">${avatarInitial}</span>
                </div>
                <div class="agent-badges">
                    ${systemType ? `<span class="system-badge system-badge-${systemType.toLowerCase()}">${systemType}</span>` : ''}
                    ${agent.is_public ? '<span class="public-badge">PUBLIC</span>' : ''}
                </div>
            </div>
            <div class="card-content">
                <h3 class="agent-name">${escapeHtml(agent.name)}</h3>
                <p class="agent-description">${escapeHtml(truncateText(agent.persona || 'AI Assistant', 80))}</p>
                <div class="agent-meta">
                    <span class="owner">by ${escapeHtml(ownerDisplay)}</span>
                    ${cloneStatHTML}
                    ${agent.capabilities ? `<span class="capabilities">${agent.capabilities.length} capabilities</span>` : ''}
                </div>
            </div>
            <div class="card-actions">
                <button class="chat-btn" onclick="event.stopPropagation(); handleChatWithAgent(event, '${agentId}')">
                    <svg width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
                        <path d="M8 1a7 7 0 0 0-7 7c0 1.933.762 3.68 2 4.95v1.55a.5.5 0 0 0 .854.354L6.707 12H8a7 7 0 1 0 0-14z"/>
                    </svg>
                    Chat
                </button>
                <button class="clone-btn" onclick="event.stopPropagation(); handleCloneAgent(event, '${agentId}')">
                    <svg width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
                        <path d="M4 2a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V2zm2-1a1 1 0 0 0-1 1v8a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1V2a1 1 0 0 0-1-1H6zM2 5a1 1 0 0 0-1 1v8a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1v-1h1v1a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h1v1H2z"/>
                    </svg>
                    Clone
                </button>
            </div>
        `;

        elements.featuredGrid.appendChild(card);
    });
}

// Fallback content if API fails
function renderMarketplaceFallback() {
    if (!elements.featuredGrid) return;

    elements.featuredGrid.innerHTML = `
        <div class="error-message" style="grid-column: 1 / -1; text-align: center; padding: 40px;">
            <h3 style="color: var(--error); margin-bottom: 16px;">Unable to load marketplace</h3>
            <p style="color: var(--text-secondary); margin-bottom: 24px;">Check your connection and try again</p>
            <button class="btn btn-primary" onclick="loadFeaturedContent()">Retry</button>
        </div>
    `;
}

// Handle agent selection from marketplace
function handleAgentSelection(agent) {
    if (!state.isAuthenticated) {
        showAuthModal();
        return;
    }

    // Show options modal for the agent
    showAgentOptionsModal(agent);
}

// NEW/IMPROVED: This modal now shows more details and is more robust.
async function showAgentOptionsModal(agent) {
    const agentId = agent.agent_id || agent.id;

    showLoading(true);
    const detailedAgent = await fetchAgentDetails(agentId);
    showLoading(false);

    if (!detailedAgent) {
        showError("Could not load agent details.");
        return;
    }

    const overlay = document.createElement('div');
    overlay.className = 'modal-overlay';
    overlay.id = 'agent-options-overlay';
    overlay.style.display = 'flex';
    overlay.onclick = (e) => { if (e.target === overlay) closeAgentOptionsModal(); };

    const modal = document.createElement('div');
    modal.className = 'modal wide'; // Use the wide class for more space

    const avatar = detailedAgent.name ? detailedAgent.name[0].toUpperCase() : 'A';
    const systemType = getSystemType(detailedAgent);
    const ownerDisplay = getOwnerDisplay(detailedAgent);
    const cloneCount = detailedAgent.clone_count || 0;
    const avatarUrl = detailedAgent.avatar_url ? `${CONFIG.API_BASE}${detailedAgent.avatar_url}` : '';

    // Generate HTML for Biographical Memories
    let memoriesHTML = '';
    // The API now sends all memories under the 'sample_memories' key for pre-built,
    // or we can fetch them for user-published agents.
    const biographicalMemories = detailedAgent.sample_memories || [];

    if (biographicalMemories.length > 0) {
        memoriesHTML = `
            <div class="modal-section">
                <h4>Biographical Memories (${biographicalMemories.length})</h4>
                <div class="modal-memories-container">
                    ${biographicalMemories.map(mem => `
                        <div class="memory-preview-item">
                            <span class="memory-type-badge">${mem.type}</span>
                            <p>"${escapeHtml(mem.content)}"</p>
                        </div>
                    `).join('')}
                </div>
            </div>`;
    }

    // Generate HTML for Core Identity (for CEAF agents)
    let identityHTML = '';
    if (systemType === 'CEAF' && detailedAgent.detailed_persona) {
         identityHTML = `
            <div class="modal-section">
                <h4>Core Identity (NCIM)</h4>
                <div class="identity-preview-item">
                    <p>${escapeHtml(detailedAgent.detailed_persona)}</p>
                </div>
            </div>`;
    }

    modal.innerHTML = `
        <div class="modal-header">
            <h2>${escapeHtml(detailedAgent.name)}</h2>
            <button class="modal-close" onclick="closeAgentOptionsModal()">&times;</button>
        </div>
        <div class="modal-content">
            <div class="modal-agent-header">
                <div class="agent-avatar" style="width: 80px; height: 80px; font-size: 32px;">
                    <img src="${avatarUrl}" alt="${detailedAgent.name} Avatar" style="${!avatarUrl && 'display:none;'}" onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';">
                    <span style="${avatarUrl && 'display:none;'}">${avatar}</span>
                </div>
                <div class="modal-agent-info">
                    <p class="modal-agent-persona">${escapeHtml(detailedAgent.persona || 'AI Assistant')}</p>
                    <div class="modal-agent-meta">
                        <span class="system-badge system-badge-${systemType.toLowerCase()}">${systemType}</span>
                        <span class="owner-badge">by ${escapeHtml(ownerDisplay)}</span>
                        <span class="clone-badge" title="${cloneCount} Clones">
                            <svg width="14" height="14" viewBox="0 0 16 16"><path d="M4 2a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V2zm2-1a1 1 0 0 0-1 1v8a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1V2a1 1 0 0 0-1-1H6zM2 5a1 1 0 0 0-1 1v8a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1v-1h1v1a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h1v1H2z"/></svg>
                            ${cloneCount}
                        </span>
                    </div>
                </div>
            </div>

            ${identityHTML}
            ${memoriesHTML}

            <div class="modal-actions">
                <button class="btn-auth" style="flex:1;" onclick="startChatWithAgent('${agentId}')">Start Chatting</button>
                <button class="clone-btn" style="flex:1;" onclick="cloneAgentToLibrary('${agentId}')">Clone to Library</button>
            </div>
        </div>
    `;

    overlay.appendChild(modal);
    document.body.appendChild(overlay);
}

// Helper functions
function getSystemType(agent) {
    if (agent.settings && agent.settings.system_type) {
        return agent.settings.system_type.toUpperCase();
    }
    if (agent.capabilities) {
        const hasCEAF = agent.capabilities.some(cap => cap.toLowerCase().includes('ceaf') || cap.includes('adaptive_memory'));
        if (hasCEAF) return 'CEAF';
    }
    return 'NCF'; // Default to NCF
}

function getOwnerDisplay(agent) {
    return agent.owner_username || 'Public';
}

function truncateText(text, maxLength) {
    if (text.length <= maxLength) return text;
    return text.substr(0, maxLength - 3) + '...';
}

// Action handlers
async function handleChatWithAgent(event, agentId) {
    event.stopPropagation();
    showLoading(true);

    try {
        const agentDetails = await fetchAgentDetails(agentId);
        if (!agentDetails) throw new Error("Could not retrieve agent details.");

        // Check if the current user is the owner.
        // The API provides `is_owner` for private agents, but we can also check the user_id.
        const isOwner = agentDetails.owner_id === state.currentUser?.user_id;

        if (isOwner) {
            // If the user owns this agent, just start the chat directly.
            await startChatWithAgent(agentId);
        } else {
            // If the user does NOT own this agent (i.e., it's a public template),
            // clone it first, then chat with the clone.
            console.log(`Cloning public agent ${agentId} before starting chat...`);
            const cloneResult = await cloneAgentToLibrary(agentId);

            if (cloneResult && cloneResult.agent_id) {
                // IMPORTANT: Start the chat with the NEW agent ID from the clone response.
                await startChatWithAgent(cloneResult.agent_id);
            } else {
                throw new Error("Cloning failed, cannot start chat.");
            }
        }
    } catch (error) {
        showError(error.message);
    } finally {
        showLoading(false);
    }
}

async function handleCloneAgent(event, agentId) {
    event.stopPropagation();
    await cloneAgentToLibrary(agentId);
}

async function startChatWithAgent(agentId) {
    try {
        showLoading(true);
        const agent = await fetchAgentDetails(agentId);
        if (agent) {
            await selectAgent(agent);
            closeAgentOptionsModal();
        } else {
            throw new Error("Agent details not found.");
        }
    } catch (error) {
        console.error('Error starting chat with agent:', error);
        showError('Failed to start chat with agent');
    } finally {
        showLoading(false);
    }
}


async function cloneAgentToLibrary(agentId, customName = null) {
    showLoading(true);
    try {
        const payload = { source_agent_id: agentId };
        if (customName) {
            payload.custom_name = customName;
        }

        const response = await fetch(`${CONFIG.API_BASE}/agents/clone`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('aura_token')}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        const result = await response.json();
        if (!response.ok) {
            throw new Error(result.detail || 'Failed to clone agent');
        }

        showSuccess(`Agent cloned as "${result.name || 'your new agent'}"!`);
        await loadMyAgents(); // Refresh the list of user's agents
        await loadRecentChats();
        return result; // Return the new agent's data

    } catch (error) {
        console.error('Error cloning agent:', error);
        showError(error.message);
        return null; // Return null on failure
    } finally {
        showLoading(false);
        closeAgentOptionsModal();
    }
}

async function fetchAgentDetails(agentId) {
    try {
        const token = localStorage.getItem('aura_token');
        const headers = token ? { 'Authorization': `Bearer ${token}` } : {};

        // Try fetching from the detailed prebuilt endpoint first
        const prebuiltResponse = await fetch(`${CONFIG.API_BASE}/prebuilt-agents/${agentId}`, { headers });
        if (prebuiltResponse.ok) {
            const agentData = await prebuiltResponse.json();
            // Standardize the ID field for consistency
            agentData.agent_id = agentData.id;
            return agentData;
        }

        // Fallback to the generic agent detail endpoint
        const genericResponse = await fetch(`${CONFIG.API_BASE}/agents/${agentId}`, { headers });
        if (genericResponse.ok) {
            return await genericResponse.json();
        }

        return null;
    } catch (error) {
        console.error('Error fetching agent details:', error);
        return null;
    }
}

function closeAgentOptionsModal() {
    const overlay = document.getElementById('agent-options-overlay');
    if (overlay) {
        overlay.remove();
    }
    document.body.style.overflow = 'auto';
}

function updateMarketplaceTitle() {
    const featuredSection = document.querySelector('.featured-agents h2');
    if (featuredSection) {
        featuredSection.innerHTML = `
            <span>🏪 Agent Marketplace</span>
            <button class="btn btn-ghost" onclick="loadFeaturedContent()" style="margin-left: 12px; font-size: 14px;" title="Refresh marketplace">
                <svg width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
                    <path d="M11.534 7h3.932a.25.25 0 0 1 .192.41l-1.966 2.36a.25.25 0 0 1-.384 0l-1.966-2.36a.25.25 0 0 1 .192-.41zm-11 2h3.932a.25.25 0 0 0 .192-.41L2.692 6.23a.25.25 0 0 0-.384 0L.342 8.59A.25.25 0 0 0 .534 9z"/>
                    <path fill-rule="evenodd" d="M8 3c-1.552 0-2.94.707-3.857 1.818a.5.5 0 1 1-.771-.636A6.002 6.002 0 0 1 13.917 7H12.9A5.002 5.002 0 0 0 8 3zM3.1 9a5.002 5.002 0 0 0 8.757 2.182.5.5 0 1 1 .771.636A6.002 6.002 0 0 1 2.083 9H3.1z"/>
                </svg>
            </button>
        `;
    }
}

function initializeMarketplace() {
    updateMarketplaceTitle();
    loadFeaturedContent();
}




function switchView(view) {
    console.log('Switching to view:', view);
    state.currentView = view;

    // Hide all other views
    document.querySelectorAll('.view').forEach(v => {
        v.style.display = 'none';
    });

    // Reset all navigation tabs
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.remove('active');
    });

    // Show the selected view and set the correct tab to active
    switch (view) {
        case 'discover':
            if (elements.discoverView) {
                elements.discoverView.style.display = 'block';
                elements.discoverTab?.classList.add('active');
                loadFeaturedContent();
            } else {
                console.error('Discover view element not found');
            }
            break;

        case 'chat':
            if (elements.chatView) {
                elements.chatView.style.display = 'block';
            } else {
                console.error('Chat view element not found');
            }
            break;

        case 'create':
            if (elements.createView) {
                elements.createView.style.display = 'block';
                loadCreateView();
            } else {
                console.error('Create view element not found');
            }
            break;

        case 'my-agents':
            console.log('Switching to my-agents view...');
            // 1. Always activate the "My Agents" tab
            elements.myAgentsTab?.classList.add('active');

            // 2. Find and display the "My Agents" view container
            const myAgentsViewElement = elements.myAgentsView || document.getElementById('my-agents-view');

            if (myAgentsViewElement) {
                myAgentsViewElement.style.display = 'block';

                // 3. Decide what content to show INSIDE the view based on authentication
                if (state.isAuthenticated) {
                    // If logged in, load the agents
                    console.log('User is authenticated, loading agents.');
                    loadMyAgents();
                } else {
                    // If not logged in, show a sign-in prompt within the view
                    console.log('User not authenticated, showing login prompt inside view.');
                    const myAgentsGrid = document.getElementById('my-agents-grid');
                    if (myAgentsGrid) {
                        myAgentsGrid.innerHTML = `
                            <div class="auth-required-message" style="text-align: center; padding: 60px 20px;">
                                <div style="font-size: 48px; margin-bottom: 20px;">🔒</div>
                                <h3 style="color: var(--text-secondary); margin-bottom: 16px;">Authentication Required</h3>
                                <p style="color: var(--text-tertiary); margin-bottom: 24px;">Please sign in to view and manage your agents.</p>
                                <button class="btn-auth" onclick="showAuthModal()">Sign In</button>
                            </div>
                        `;
                    }
                }
            } else {
                // Fallback if the view element doesn't exist in the HTML at all
                console.error('My Agents view element not found! Using fallback.');
                createMyAgentsViewFallback();
            }
            break;

        default:
            console.error('Unknown view:', view);
    }
}

function createMyAgentsViewFallback() {
    console.log('Creating My Agents view fallback...');

    const mainContent = document.getElementById('main-content');
    if (!mainContent) {
        console.error('Main content area not found');
        return;
    }

    // Create the my-agents view
    const myAgentsView = document.createElement('div');
    myAgentsView.id = 'my-agents-view';
    myAgentsView.className = 'view';
    myAgentsView.style.display = 'block';

    myAgentsView.innerHTML = `
        <div class="view-header">
            <h1>My Agents</h1>
            <p>Manage your personal AI agents</p>
        </div>
        <div class="my-agents-content">
            <div id="my-agents-grid" class="my-agents-grid">
                <div class="spinner"></div>
            </div>
        </div>
    `;

    // Hide other views and show this one
    document.querySelectorAll('.view').forEach(v => v.style.display = 'none');
    mainContent.appendChild(myAgentsView);

    // Update elements cache
    elements.myAgentsView = myAgentsView;
    elements.myAgentsGrid = myAgentsView.querySelector('#my-agents-grid');

    // Activate the tab
    elements.myAgentsTab?.classList.add('active');

    // Load the agents
    loadMyAgents();

    console.log('My Agents view created successfully');
}


function toggleSidebar() {
    state.sidebarCollapsed = !state.sidebarCollapsed;
    elements.sidebar?.classList.toggle('collapsed', state.sidebarCollapsed);
}

function toggleUserMenu() {
    const isVisible = elements.userMenu.style.display !== 'none';
    elements.userMenu.style.display = isVisible ? 'none' : 'block';
}

// Modal functions
function showAuthModal() {
    elements.authModal.style.display = 'flex';
    document.body.style.overflow = 'hidden';
    setTimeout(() => {
        const firstInput = elements.authModal.querySelector('input');
        firstInput?.focus();
    }, 100);
}

function hideAuthModal() {
    elements.authModal.style.display = 'none';
    document.body.style.overflow = 'auto';
    clearError();
}

function handleModalOverlayClick(event) {
    if (event.target === elements.authModal) {
        hideAuthModal();
    }
}

function switchAuthTab(mode) {
    state.authMode = mode;
    elements.loginTab?.classList.toggle('active', mode === 'login');
    elements.registerTab?.classList.toggle('active', mode === 'register');
    const emailGroup = document.getElementById('email-group');
    if (emailGroup) {
        emailGroup.style.display = mode === 'register' ? 'block' : 'none';
    }
    if (elements.authSubmitBtn) {
        elements.authSubmitBtn.textContent = mode === 'login' ? 'Sign in' : 'Sign up';
    }
    const emailInput = document.getElementById('email');
    if (emailInput) {
        emailInput.required = mode === 'register';
    }
}


async function loadMyAgents() {
    console.log('Loading my agents...');

    if (!state.isAuthenticated) {
        console.log('User not authenticated');
        return;
    }

    if (!elements.myAgentsGrid) {
        console.error('My Agents Grid element not found');
        // Try to find it again
        elements.myAgentsGrid = document.getElementById('my-agents-grid') ||
                               document.querySelector('.my-agents-grid');

        if (!elements.myAgentsGrid) {
            console.error('Could not find or create my-agents-grid element');
            return;
        }
    }

    elements.myAgentsGrid.innerHTML = '<div class="spinner"></div>';
    showLoading(true);

    try {
        const response = await fetch(`${CONFIG.API_BASE}/agents/my-agents`, {
            headers: { 'Authorization': `Bearer ${localStorage.getItem('aura_token')}` }
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: Failed to load your agents`);
        }

        const myAgents = await response.json();
        console.log('Loaded agents:', myAgents.length);

        state.agents = myAgents;
        renderMyAgents(myAgents);

    } catch (error) {
        console.error('Failed to load my agents:', error);
        elements.myAgentsGrid.innerHTML = `
            <div class="error-message" style="text-align: center; padding: 40px;">
                <h3>Failed to load your agents</h3>
                <p>${error.message}</p>
                <button class="btn btn-primary" onclick="loadMyAgents()" style="margin-top: 16px;">
                    Try Again
                </button>
            </div>
        `;
    } finally {
        showLoading(false);
    }
}

function renderMyAgents(agents) {
    if (!elements.myAgentsGrid) return;
    elements.myAgentsGrid.innerHTML = '';
    if (agents.length === 0) {
        elements.myAgentsGrid.innerHTML = `<p class="placeholder-text">You haven't created any agents yet. Click "Create Agent" to start!</p>`;
        return;
    }
    agents.forEach(agent => {
        const card = document.createElement('div');
        card.className = 'my-agent-card';
        const avatarInitial = agent.name ? agent.name[0].toUpperCase() : 'A';
        const avatarUrl = agent.avatar_url ? `${CONFIG.API_BASE}${agent.avatar_url}` : '';
        const isPublicTemplate = agent.is_public_template;
        const systemType = getSystemType(agent);

        let statsHTML = '';
        let actionsHTML = '';

        if (isPublicTemplate) {
            statsHTML = `
                <div class="my-agent-card__stats">
                    <div class="stat-item"><span class="value">${agent.version}</span><span class="label">Version</span></div>
                    <div class="stat-item"><span class="value">${agent.clone_count}</span><span class="label">Clones</span></div>
                    <div class="stat-item"><span class="value">${agent.usage_cost.toFixed(2)}</span><span class="label">Price</span></div>
                </div>
            `;
            actionsHTML = `
                <div class="my-agent-card__actions">
                    <button class="btn-action" onclick="showSetPriceModal('${agent.agent_id}', '${escapeHtml(agent.name)}', ${agent.usage_cost})">Set Price</button>
                    <button class="btn-action" onclick="publishAgent('${agent.parent_agent_id}')" title="Publish an updated version from the private source agent">Update</button>
                    <button class="btn-action btn-danger" onclick="unpublishAgent('${agent.agent_id}')">Unpublish</button>
                </div>
            `;
        } else {
            actionsHTML = `
                <div class="my-agent-card__actions">
                    <button class="btn-action" onclick="selectAgentById('${agent.agent_id}')">Chat</button>
                    <button class="btn-action" onclick="showMemoryModal('${agent.agent_id}', '${escapeHtml(agent.name)}')">Memories</button>
                    <button class="btn-action" onclick="publishAgent('${agent.agent_id}')">Publish</button>
                    <button class="btn-action btn-danger" onclick="deleteAgent('${agent.agent_id}')">Delete</button>
                </div>`;
        }

        // RESTRUCTURED CARD HTML
        card.innerHTML = `
            <div class="my-agent-card__header">
                <div class="agent-avatar">
                    <img src="${avatarUrl}" alt="${agent.name} Avatar" style="${!avatarUrl && 'display:none;'}" onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';">
                    <span style="${avatarUrl && 'display:none;'}">${avatarInitial}</span>
                </div>
                <div class="my-agent-card__info">
                    <h3>${escapeHtml(agent.name)}</h3>
                    <p>${escapeHtml(agent.persona)}</p>
                </div>
            </div>

            <div class="my-agent-card__details">
                <div class="detail-item">
                    <span class="label">Type</span>
                    <span class="agent-tag ${isPublicTemplate ? 'public' : 'private'}">${isPublicTemplate ? 'Public Template' : 'Private Agent'}</span>
                </div>
                <div class="detail-item">
                    <span class="label">System</span>
                    <span class="system-badge system-badge-${systemType.toLowerCase()}">${systemType}</span>
                </div>
                <div class="detail-item">
                    <span class="label">Model</span>
                    <span class="value model-value">${escapeHtml(agent.model.split('/').pop())}</span>
                </div>
            </div>

            ${statsHTML}
            ${actionsHTML}
        `;
        elements.myAgentsGrid.appendChild(card);
    });
}








// Function to delete a private agent
async function deleteAgent(agentId) {
    if (!confirm(`Are you sure you want to permanently delete this agent? This action cannot be undone.`)) {
        return;
    }

    showLoading(true);
    try {
        const response = await fetch(`${CONFIG.API_BASE}/agents/${agentId}`, {
            method: 'DELETE',
            headers: { 'Authorization': `Bearer ${localStorage.getItem('aura_token')}` }
        });

        if (response.ok) {
            showSuccess("Agent deleted successfully.");
            await loadMyAgents(); // Refresh the list
        } else {
            const error = await response.json();
            throw new Error(error.detail || "Failed to delete agent.");
        }
    } catch (error) {
        showError(error.message);
    } finally {
        showLoading(false);
    }
}

// Function to unpublish an agent
async function unpublishAgent(agentId) {
    if (!confirm("Are you sure you want to unpublish this agent? It will be removed from the marketplace.")) {
        return;
    }

    showLoading(true);
    try {
        const response = await fetch(`${CONFIG.API_BASE}/agents/template/${agentId}/unpublish`, {
            method: 'DELETE',
            headers: { 'Authorization': `Bearer ${localStorage.getItem('aura_token')}` }
        });

        if (response.ok) {
            showSuccess("Agent unpublished successfully.");
            await loadMyAgents(); // Refresh the list
        } else {
            const error = await response.json();
            throw new Error(error.detail || "Failed to unpublish agent.");
        }
    } catch (error) {
        showError(error.message);
    } finally {
        showLoading(false);
    }
}

// Functions for the price modal
function showSetPriceModal(agentId, agentName, currentPrice) {
    state.currentEditingAgentId = agentId;
    elements.priceAgentName.textContent = agentName;
    elements.agentPriceInput.value = currentPrice;
    elements.setPriceModal.style.display = 'flex';
    document.body.style.overflow = 'hidden';
}

function closeSetPriceModal() {
    state.currentEditingAgentId = null;
    elements.setPriceModal.style.display = 'none';
    document.body.style.overflow = 'auto';
}

async function saveAgentPrice() {
    const agentId = state.currentEditingAgentId;
    const newPrice = parseFloat(elements.agentPriceInput.value);

    if (isNaN(newPrice) || newPrice < 0) {
        showError("Please enter a valid, non-negative price.");
        return;
    }

    showLoading(true);
    try {
        const response = await fetch(`${CONFIG.API_BASE}/agents/template/${agentId}/price`, {
            method: 'PUT',
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('aura_token')}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ usage_cost: newPrice })
        });

        if (response.ok) {
            showSuccess("Agent price updated successfully.");
            closeSetPriceModal();
            await loadMyAgents(); // Refresh the view
        } else {
            const error = await response.json();
            throw new Error(error.detail || "Failed to set price.");
        }
    } catch (error) {
        showError(error.message);
    } finally {
        showLoading(false);
    }
}

// Agent management
// WITH THIS NEW FUNCTION
async function loadRecentChats() {
    if (!state.isAuthenticated) return;

    try {
        const response = await fetch(`${CONFIG.API_BASE}/chat/sessions?limit=20`, {
            headers: { 'Authorization': `Bearer ${localStorage.getItem('aura_token')}` }
        });

        if (response.ok) {
            const data = await response.json();
            // Store agent details from sessions for easy access
            state.agents = data.sessions.map(session => ({
                agent_id: session.agent_id,
                name: session.agent_name,
            }));
            renderRecentChats(data.sessions);
        }
    } catch (error) {
        console.error('Failed to load recent chats:', error);
    }
}

function renderRecentChats(sessions) {
    if (!elements.agentsList) return;
    elements.agentsList.innerHTML = '';

    if (!sessions || sessions.length === 0) {
        elements.agentsList.innerHTML = `
            <div style="text-align: center; color: var(--text-muted); padding: 20px; font-size: 14px;">
                No recent chats. Start a new one!
            </div>
        `;
        elements.agentCount.textContent = '0';
        return;
    }

    // Use a Set to only show one entry per agent, using the most recent session
    const uniqueAgentIds = new Set();
    const uniqueSessions = sessions.filter(session => {
        if (uniqueAgentIds.has(session.agent_id)) {
            return false;
        }
        uniqueAgentIds.add(session.agent_id);
        return true;
    });

    uniqueSessions.forEach(session => {
        const agentForDisplay = {
            agent_id: session.agent_id,
            name: session.agent_name,
            avatar_url: null // Session endpoint doesn't include avatar
        };
        const sessionElement = createAgentListItem(agentForDisplay);
        elements.agentsList.appendChild(sessionElement);
    });

    elements.agentCount.textContent = uniqueSessions.length.toString();
}

function renderAgentFiles(files) {
    if (files.length === 0) {
        elements.filesListContainer.innerHTML = `<p class="placeholder-text">No knowledge files uploaded yet.</p>`;
        return;
    }
    elements.filesListContainer.innerHTML = files.map(file => `
        <div class="file-item">
            <div class="file-info">
                <span>📄</span>
                <span>${escapeHtml(file.filename)}</span>
            </div>
            <div class="file-size">${(file.size / 1024).toFixed(1)} KB</div>
        </div>
    `).join('');
}

function updateAvatarDisplay(imgElement, spanElement, agent) {
    const avatarUrl = agent.avatar_url ? `${CONFIG.API_BASE}${agent.avatar_url}` : '';
    if (avatarUrl) {
        imgElement.src = avatarUrl;
        imgElement.style.display = 'block';
        spanElement.style.display = 'none';
    } else {
        spanElement.textContent = agent.name ? agent.name[0].toUpperCase() : 'A';
        imgElement.style.display = 'none';
        spanElement.style.display = 'flex';
    }
}

function updateAllAvatarDisplays(agent) {
    // Chat Header
    updateAvatarDisplay(
        elements.currentAgentAvatar.querySelector('img'),
        elements.currentAgentAvatar.querySelector('span'),
        agent
    );
    // Profile Modal
    updateAvatarDisplay(
        elements.profileModalAvatarImg,
        elements.profileModalAvatarInitial,
        agent
    );
    // Sidebar list
    const sidebarItem = document.querySelector(`#agent-list-item-${agent.agent_id} .agent-avatar`);
    if(sidebarItem) {
         updateAvatarDisplay(
            sidebarItem.querySelector('img'),
            sidebarItem.querySelector('span'),
            agent
        );
    }
   // Marketplace (less direct, might need a full re-render or more complex update)
   // For simplicity, we can just re-render the marketplace
   if(state.currentView === 'discover'){
       initializeMarketplace();
   }
}

async function handleKnowledgeFileUpload(event) {
    const file = event.target.files[0];
    if (!file || !state.currentAgent) return;

    showLoading(true);
    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${CONFIG.API_BASE}/agents/${state.currentAgent.agent_id}/files`, {
            method: 'POST',
            headers: { 'Authorization': `Bearer ${localStorage.getItem('aura_token')}` },
            body: formData,
        });

        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || 'File upload failed');

        showSuccess(result.message);
        await loadAgentFiles(); // Refresh the list
    } catch (error) {
        showError(error.message);
    } finally {
        showLoading(false);
        elements.knowledgeFileInput.value = ''; // Reset input
    }
}


async function selectAgentById(agentId) {

    let agent = state.agents.find(a => a.agent_id === agentId);


    if (!agent) {
        console.warn(`Agent ${agentId} not found in local state. Fetching from API.`);
        try {
            agent = await fetchAgentDetails(agentId); // This function fetches full agent data.
            if (agent) {
                // Optional but good practice: update the local state.
                const existingAgentIndex = state.agents.findIndex(a => a.agent_id === agentId);
                if (existingAgentIndex > -1) {
                    state.agents[existingAgentIndex] = agent;
                } else {
                    state.agents.push(agent);
                }
            }
        } catch (fetchError) {
             console.error(`Failed to fetch details for agent ${agentId}:`, fetchError);
             showError("Could not load agent details. Please refresh the page.");
             return; // Stop execution if fetch fails
        }
    }

    if (agent) {
        await selectAgent(agent);
    } else {
        console.error(`Agent with ID ${agentId} could not be found.`);
        showError("Could not find agent details. Please refresh the page.");
    }
}

function createAgentListItem(agent) {
    const item = document.createElement('div');
    item.className = 'agent-item';
    item.id = `agent-list-item-${agent.agent_id}`;
    const avatarInitial = agent.name ? agent.name[0].toUpperCase() : 'A';
    const avatarUrl = agent.avatar_url ? `${CONFIG.API_BASE}${agent.avatar_url}` : '';

    item.innerHTML = `
        <div class="agent-avatar" onclick="selectAgentById('${agent.agent_id}')">
            <img src="${avatarUrl}" alt="${agent.name} Avatar" style="${!avatarUrl && 'display:none;'}" onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';">
            <span style="${avatarUrl && 'display:none;'}">${avatarInitial}</span>
        </div>
        <div class="agent-name" onclick="selectAgentById('${agent.agent_id}')">${escapeHtml(agent.name)}</div>
        <div class="agent-item-actions">
             <button class="agent-item-btn" onclick="event.stopPropagation(); showMemoryModal('${agent.agent_id}', '${escapeHtml(agent.name)}')" title="Memory Management">🧠</button>
             <!-- O BOTÃO DE PUBLICAR FOI REMOVIDO DAQUI -->
        </div>
    `;
    return item;
}

function publishAgent(agentId) {
    if (!agentId) {
        showError("No agent selected to publish.");
        return;
    }
    const agent = state.agents.find(a => a.agent_id === agentId);
    if (!agent) {
        showError("Could not find agent to publish.");
        return;
    }
    state.agentToPublish = agentId;
    elements.publishAgentName.textContent = agent.name;
    elements.publishChangelog.value = ''; // Limpa o campo
    elements.publishModal.style.display = 'flex';
    document.body.style.overflow = 'hidden';
    toggleAgentDropdown(); // Fecha o menu do chat
}

function closePublishModal() {
    elements.publishModal.style.display = 'none';
    document.body.style.overflow = 'auto';
    state.agentToPublish = null;
}

async function confirmPublishAgent() {
    const agentId = state.agentToPublish;
    const changelog = elements.publishChangelog.value.trim();
    const includeHistory = elements.publishIncludeHistory.checked;

    if (!agentId) return;
    if (!changelog) {
        showError("Please provide version notes (changelog) for this publication.");
        return;
    }

    elements.confirmPublishBtn.disabled = true;
    elements.confirmPublishBtn.textContent = 'Publishing...';
    showLoading(true);

    try {
        const formData = new FormData();
        formData.append('changelog', changelog);
        formData.append('include_chat_history', includeHistory); // Envia o valor do toggle

        const response = await fetch(`${CONFIG.API_BASE}/agents/${agentId}/publish`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('aura_token')}`
            },
            body: formData
        });

        const result = await response.json();
        if (!response.ok) {
            throw new Error(result.detail || 'Failed to publish agent.');
        }

        showSuccess(result.message);
        closePublishModal();
        await loadFeaturedContent(); // Atualiza o marketplace

    } catch (error) {
        console.error("Publish error:", error);
        showError(error.message);
    } finally {
        showLoading(false);
        elements.confirmPublishBtn.disabled = false;
        elements.confirmPublishBtn.textContent = 'Publish Version 1.0.0';
    }
}

async function selectAgent(agent) {
    if (state.currentAgent && state.currentAgent.agent_id === agent.agent_id && state.currentView === 'chat') {
        return;
    }

    showLoading(true);
    const detailedAgent = await fetchAgentDetails(agent.agent_id);
    showLoading(false);

    if (!detailedAgent) {
        showError("Could not load the selected agent's full profile.");
        return;
    }

    state.currentAgent = detailedAgent;
    state.sessionId = null; // Reset session ID, we will get the correct one for saving shortly

    state.sessionModelOverride = null;
    state.currentPrecisionMode = false;

    // 1. Setup the UI immediately
    switchView('chat');
    updateChatHeader();
    clearMessages();
    addMessage('assistant', 'Connecting and restoring your conversation...');

    // 2. Perform two parallel but distinct tasks:
    //    a) Get a session ID for SAVING NEW messages.
    //    b) Load the ENTIRE history for DISPLAY.
    await populateChatModelSelector();
    await getOrCreateSession(agent.agent_id); // This just sets state.sessionId for future sends.
    await loadChatHistory(agent.agent_id);   // This now loads the full history using the new endpoint.


    // Update the visual selection in the sidebar
    document.querySelectorAll('.agent-item').forEach(item => item.classList.remove('active'));
    const activeItem = document.getElementById(`agent-list-item-${agent.agent_id}`);
    if (activeItem) {
        activeItem.classList.add('active');
    }
}

async function updateChatHeader() {
    if (!state.currentAgent) return;
    const agent = state.currentAgent;

    updateAvatarDisplay(
        elements.currentAgentAvatar.querySelector('img'),
        elements.currentAgentAvatar.querySelector('span'),
        agent
    );

    updatePrecisionModeIndicator();

    if (elements.chatModelSelector) {
        elements.chatModelSelector.disabled = false; // Always enabled
        elements.chatModelSelector.parentElement.style.display = 'flex'; // Always show
    }

    if (elements.currentAgentName) {
        elements.currentAgentName.textContent = agent.name;
    }
    if (elements.currentAgentDescription) {
        elements.currentAgentDescription.textContent = agent.persona || 'AI Assistant';
    }

    const isOwner = state.currentAgent.is_owner;

    // ================= START: NEW HEADER UPDATE LOGIC =================
    // 1. Update Specialist Connections Display
    if (elements.connectionsDisplay) {
        elements.connectionsDisplay.innerHTML = `
            <div class="connection-icon" title="Image Generation Specialist">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg>
            </div>
            <div class="connection-icon" title="Speech Synthesis Specialist">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path><path d="M19 10v2a7 7 0 0 1-14 0v-2"></path><line x1="12" y1="19" x2="12" y2="23"></line><line x1="8" y1="23" x2="16" y2="23"></line></svg>
            </div>
        `;
    }

    // 2. Fetch and Update Live Memory Toggles
    try {
        const response = await fetch(`${CONFIG.API_BASE}/agents/${agent.agent_id}/profile`, {
            headers: { 'Authorization': `Bearer ${localStorage.getItem('aura_token')}` }
        });
        if (!response.ok) throw new Error('Could not fetch agent profile for permissions');
        const profile = await response.json();

        // The backend `AgentConfig` holds these booleans directly
        const canContribute = profile.allow_live_memory_contribution || false;
        const canInfluence = profile.allow_live_memory_influence || false;

        if (elements.liveMemoryShareToggle) {
            elements.liveMemoryShareToggle.checked = canContribute;
            elements.liveMemoryShareToggle.disabled = false;
        }
        if (elements.liveMemoryNavigateToggle) {
            elements.liveMemoryNavigateToggle.checked = canInfluence;
            elements.liveMemoryNavigateToggle.disabled = false;
        }
    } catch (error) {
        console.error("Error fetching live memory permissions:", error);
        // Disable toggles on error
        if (elements.liveMemoryShareToggle) elements.liveMemoryShareToggle.disabled = true;
        if (elements.liveMemoryNavigateToggle) elements.liveMemoryNavigateToggle.disabled = true;
    }
    // ================== END: NEW HEADER UPDATE LOGIC ==================

    // ================= START: NEW WIDGET UPDATE LOGIC =================
    closeAllMindPopovers(); // Close any open popovers when switching agents
    const systemType = getSystemType(agent);

    // Show/hide CEAF-only widgets
    document.querySelectorAll('.ceaf-only').forEach(el => {
        el.style.display = (systemType === 'CEAF') ? 'flex' : 'none';
    });

    // Fetch data for visible widgets
    fetchRecentMemories(agent.agent_id);
    if (systemType === 'CEAF') {
        fetchCeafStatus(agent.agent_id);
    }
    // ================== END: NEW WIDGET UPDATE LOGIC ==================

}

function toggleAgentDropdown() {
    const isVisible = elements.agentDropdownMenu.style.display === 'block';
    elements.agentDropdownMenu.style.display = isVisible ? 'none' : 'block';
}


function closeAllMindPopovers() {
    document.querySelectorAll('.mind-popover').forEach(popover => {
        popover.classList.remove('visible');
    });
}

function toggleMindPopover(event, popoverId) {
    event.stopPropagation();
    const targetPopover = document.getElementById(popoverId);
    const isVisible = targetPopover.classList.contains('visible');

    // Close all popovers first
    closeAllMindPopovers();

    // If it wasn't visible, show it
    if (!isVisible) {
        targetPopover.classList.add('visible');

        // Add close functionality to its close button
        const closeBtn = targetPopover.querySelector('.popover-close');
        closeBtn.onclick = () => {
            targetPopover.classList.remove('visible');
        };
    }
}

async function fetchRecentMemories(agentId) {
    const popoverContent = document.getElementById('memory-popover-content');
    popoverContent.innerHTML = `<div class="spinner small"></div>`;
    try {
        const response = await fetch(`${CONFIG.API_BASE}/agents/${agentId}/recent-memories`, {
            headers: { 'Authorization': `Bearer ${localStorage.getItem('aura_token')}` }
        });
        if (!response.ok) throw new Error('Could not fetch memories');
        const memories = await response.json();
        renderRecentMemories(memories, popoverContent);
    } catch (error) {
        console.error("Error fetching recent memories:", error);
        popoverContent.innerHTML = `<p class="error-message" style="margin:0;">${error.message}</p>`;
    }
}

async function fetchCeafStatus(agentId) {
    const statePopoverContent = document.getElementById('state-popover-content');
    const identityPopoverContent = document.getElementById('identity-popover-content');
    statePopoverContent.innerHTML = `<div class="spinner small"></div>`;
    identityPopoverContent.innerHTML = `<div class="spinner small"></div>`;

    try {
        const response = await fetch(`${CONFIG.API_BASE}/agents/${agentId}/ceaf-status`, {
            headers: { 'Authorization': `Bearer ${localStorage.getItem('aura_token')}` }
        });
        if (!response.ok) throw new Error('Could not fetch CEAF status');
        const status = await response.json();
        renderCeafStatus(status, statePopoverContent, identityPopoverContent);
    } catch (error) {
        console.error("Error fetching CEAF status:", error);
        statePopoverContent.innerHTML = `<p class="error-message" style="margin:0;">${error.message}</p>`;
        identityPopoverContent.innerHTML = `<p class="error-message" style="margin:0;">${error.message}</p>`;
    }
}

function renderRecentMemories(memories, container) {
    if (!memories || memories.length === 0) {
        container.innerHTML = `<p class="placeholder-text" style="padding:10px;">No recent memories found.</p>`;
        return;
    }
    container.innerHTML = memories.map(mem => `
        <div class="memory-item-popover">
            <span class="memory-type-badge">${mem.memory_type}</span>
            <p>"${escapeHtml(truncateText(mem.content, 120))}"</p>
        </div>
    `).join('');
}

function renderCeafStatus(status, stateContainer, identityContainer) {
    // Render Coherence State
    stateContainer.innerHTML = `
        <div class="coherence-state-display">
            <div class="state-value">${status.mcl_state.replace(/_/g, ' ')}</div>
        </div>
    `;

    // Render Identity Evolution
    identityContainer.innerHTML = `
        <div class="identity-evolution-display">
            <div class="identity-part">
                <strong>Original:</strong>
                <p>"${escapeHtml(truncateText(status.identity_evolution.original, 150))}"</p>
            </div>
            <div class="identity-part">
                <strong>Current:</strong>
                <p>"${escapeHtml(truncateText(status.identity_evolution.current, 150))}"</p>
            </div>
        </div>
    `;
}


// --- Agent Profile Modal ---
function showProfileModal() {
    if (!state.currentAgent) return;
    const agent = state.currentAgent;

    elements.profileModalAgentName.textContent = agent.name;
    updateAvatarDisplay(elements.profileModalAvatarImg, elements.profileModalAvatarInitial, agent);

    elements.profileModal.style.display = 'flex';
    document.body.style.overflow = 'hidden';
    elements.agentDropdownMenu.style.display = 'none'; // Close dropdown
}

function closeProfileModal() {
    elements.profileModal.style.display = 'none';
    document.body.style.overflow = 'auto';
}

async function handleAvatarUpload(event) {
    const file = event.target.files[0];
    if (!file || !state.currentAgent) return;

    showLoading(true);
    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${CONFIG.API_BASE}/agents/${state.currentAgent.agent_id}/avatar`, {
            method: 'POST',
            headers: { 'Authorization': `Bearer ${localStorage.getItem('aura_token')}` },
            body: formData,
        });

        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || 'Upload failed');

        // Update local state and UI
        state.currentAgent.avatar_url = result.avatar_url;
        const agentInList = state.agents.find(a => a.agent_id === state.currentAgent.agent_id);
        if(agentInList) agentInList.avatar_url = result.avatar_url;

        // Refresh all avatar displays for this agent
        updateAllAvatarDisplays(state.currentAgent);
        showSuccess('Avatar updated successfully!');
        closeProfileModal();
    } catch (error) {
        showError(error.message);
    } finally {
        showLoading(false);
        elements.avatarUploadInput.value = ''; // Reset file input
    }
}

// --- Agent Files Modal ---
async function showFilesModal() {
    if (!state.currentAgent) return;
    elements.filesModal.style.display = 'flex';
    document.body.style.overflow = 'hidden';
    elements.agentDropdownMenu.style.display = 'none'; // Close dropdown
    await loadAgentFiles();
}

function closeFilesModal() {
    elements.filesModal.style.display = 'none';
    document.body.style.overflow = 'auto';
}

async function loadAgentFiles() {
    elements.filesListContainer.innerHTML = '<div class="spinner"></div>';
    try {
        const response = await fetch(`${CONFIG.API_BASE}/agents/${state.currentAgent.agent_id}/files`, {
            headers: { 'Authorization': `Bearer ${localStorage.getItem('aura_token')}` },
        });
        if (!response.ok) throw new Error('Could not fetch files.');

        const files = await response.json();
        renderAgentFiles(files);
    } catch (error) {
        elements.filesListContainer.innerHTML = `<p class="placeholder-text error-message">${error.message}</p>`;
    }
}



// ================= START: NEW LIVE MEMORY HANDLER =================
async function handleLiveMemoryToggle(event) {
    if (!state.currentAgent) return;

    const shareToggle = elements.liveMemoryShareToggle;
    const navigateToggle = elements.liveMemoryNavigateToggle;

    // Disable toggles to prevent spamming
    shareToggle.disabled = true;
    navigateToggle.disabled = true;

    const payload = {
        allow_contribution: shareToggle.checked,
        allow_influence: navigateToggle.checked
    };

    try {
        const response = await fetch(`${CONFIG.API_BASE}/agents/${state.currentAgent.agent_id}/live-memory-permissions`, {
            method: 'PUT',
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('aura_token')}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to update permissions');
        }

        const result = await response.json();
        showSuccess('Live Memory settings updated!');

        // Update the agent state locally
        state.currentAgent.allow_live_memory_contribution = result.contribution_enabled;
        state.currentAgent.allow_live_memory_influence = result.influence_enabled;

    } catch (error) {
        console.error("Live Memory Toggle Error:", error);
        showError(error.message);
        // Revert UI to the actual state on failure
        shareToggle.checked = !shareToggle.checked;
        navigateToggle.checked = !navigateToggle.checked;
    } finally {
        // Re-enable toggles
        shareToggle.disabled = false;
        navigateToggle.disabled = false;
    }
}
// ================== END: NEW LIVE MEMORY HANDLER ==================

async function populateChatModelSelector() {
    if (!elements.chatModelSelector) return;

    elements.chatModelSelector.innerHTML = '<option>Loading models...</option>';
    try {
        const response = await fetch(`${CONFIG.API_BASE}/models/openrouter`);
        if (!response.ok) throw new Error('Failed to fetch models');

        const models = await response.json();
        elements.chatModelSelector.innerHTML = ''; // Clear loading text

        Object.entries(models).forEach(([category, modelList]) => {
            const optgroup = document.createElement('optgroup');
            optgroup.label = category;
            modelList.forEach(modelObject => { // Renomeado para clareza
                const option = document.createElement('option');
                option.value = modelObject.name; // <-- CORRIGIDO
                option.textContent = `${modelObject.name} - ${modelObject.cost} credits`; // <-- CORRIGIDO
                optgroup.appendChild(option);
            });
            elements.chatModelSelector.appendChild(optgroup);
        });

        // Set the agent's current model as selected
        if (state.currentAgent && state.currentAgent.model) {
            elements.chatModelSelector.value = state.currentAgent.model;
        }

    } catch (error) {
        console.error('Error populating model selector:', error);
        elements.chatModelSelector.innerHTML = '<option>Error loading</option>';
    }
}

async function updateAgentModel() {
    const newModel = elements.chatModelSelector.value;
    const agentDefaultModel = state.currentAgent.model;

    if (!newModel) return;

    // If the user selects the agent's default model, clear the override.
    // Otherwise, set the override.
    if (newModel === agentDefaultModel) {
        state.sessionModelOverride = null;
    } else {
        state.sessionModelOverride = newModel;
    }

    // Add a system message to inform the user of the change for this session
    addMessage('system', `🔄 Model for this session is now set to ${newModel}.`);
    showSuccess(`Model for this chat set to ${newModel}`);
}

async function getOrCreateSession(agentId) {
    try {
        const response = await fetch(`${CONFIG.API_BASE}/chat/get-or-create-session?agent_id=${agentId}`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('aura_token')}`,
                'Content-Type': 'application/json'
            }
        });

        if (response.ok) {
            const data = await response.json();
            state.sessionId = data.session_id;

            console.log(`Session established for agent ${agentId}: ${state.sessionId}`);
            return data.session_id;
        } else {
            const error = await response.json();
            console.error("Failed to get or create session:", error.detail);
            addMessage('system', `Error creating session: ${error.detail}`);
            return null;
        }
    } catch (error) {
        console.error("Error establishing session:", error);
        addMessage('system', 'Network error: Could not establish a chat session.');
        return null;
    }
}

async function loadChatHistory(agentId) {
    try {
        // This function now uses the new AGENT-based history endpoint.
        // It no longer needs a session ID to run.

        const response = await fetch(`${CONFIG.API_BASE}/chat/history/agent/${agentId}`, {
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('aura_token')}`
            }
        });

        clearMessages(); // Clear the "Connecting..." message

        if (response.ok) {
            const data = await response.json();

            if (data.messages && data.messages.length > 0) {
                // History exists, render it.
                state.messages = data.messages.map(msg => ({
                    role: msg.role,
                    content: msg.content,
                    timestamp: new Date(msg.created_at)
                }));
                renderMessages();
            } else {
                // No history, show the welcome message.
                const agent = state.currentAgent;
                if (agent) {
                    addMessage('assistant', `Hello! I'm ${agent.name}. How can I help you today?`);
                }
            }
        } else {
            console.error("Failed to load unified chat history from server");
            addMessage('system', 'Could not load chat history.');
        }
    } catch (error) {
        console.error('Failed to load unified chat history:', error);
        clearMessages();
        addMessage('system', `Error loading history: ${error.message}`);
    }
}

function handleMessageKeyPress(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

function handleInputChange() {
    const hasContent = elements.messageInput.value.trim().length > 0;
    elements.sendBtn.disabled = !hasContent || !state.currentAgent || state.isTyping;
}

async function sendMessage() {
    const message = elements.messageInput.value.trim();
    if (!message || !state.currentAgent || state.isTyping) return;

    // Add user message to UI
    addMessage('user', message);
    elements.messageInput.value = '';
    handleInputChange();

    // Show typing indicator
    showTypingIndicator();

    try {
        // Ensure we have a session
        if (!state.sessionId) {
            await getOrCreateSession(state.currentAgent.agent_id);
        }

         const payload = {
            message: message,
            session_id: state.sessionId,
            session_overrides: {}
        };

        if (state.sessionModelOverride) {
            payload.session_overrides.model = state.sessionModelOverride;
        }


        // Use the same endpoint structure as old frontend
        const response = await fetch(`${CONFIG.API_BASE}/agents/${state.currentAgent.agent_id}/chat`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('aura_token')}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload)
        });

        if (response.ok) {
            const result = await response.json();

            // Update session ID if server returns a new one

            if (result.session_id && result.session_id !== state.sessionId) {
                state.sessionId = result.session_id;
                console.log(`Session ID updated: ${state.sessionId}`);
            }

            if (typeof result.precision_mode === 'boolean') {
                state.currentPrecisionMode = result.precision_mode;
                updatePrecisionModeIndicator(); // Atualiza a UI
            }

            // Remove typing indicator and add response
            hideTypingIndicator();
            addMessage('assistant', result.response);

        } else {
            hideTypingIndicator();
            const error = await response.json();
            addMessage('system', `Error: ${error.detail || error.error || 'Failed to send message'}`);
        }
    } catch (error) {
        console.error('Chat error:', error);
        hideTypingIndicator();
        addMessage('system', 'Network error. Please try again.');
    }
}

function addMessage(role, content) {
    const message = { role, content, timestamp: new Date() };
    state.messages.push(message);
    renderMessage(message);
    scrollToBottom();
}

function renderMessages() {
    if (!elements.chatMessages) return;

    elements.chatMessages.innerHTML = '';
    state.messages.forEach(message => renderMessage(message));
    scrollToBottom();
}

function renderMessage(message) {
    if (!elements.chatMessages) return;

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${message.role}`;

    const avatar = message.role === 'user'
        ? (state.currentUser?.username?.[0]?.toUpperCase() || 'U')
        : (state.currentAgent?.name?.[0]?.toUpperCase() || 'A');

    const isSystem = message.role === 'system';

    messageDiv.innerHTML = `
        ${!isSystem ? `<div class="agent-avatar"><span>${avatar}</span></div>` : ''}
        <div class="message-content">${escapeHtml(message.content).replace(/\n/g, '<br>')}</div>
    `;

    if (isSystem) {
        messageDiv.style.opacity = '0.8';
        messageDiv.style.fontStyle = 'italic';
        messageDiv.style.justifyContent = 'center';
    }

    elements.chatMessages.appendChild(messageDiv);
}

function showTypingIndicator() {
    state.isTyping = true;
    handleInputChange();

    const typingDiv = document.createElement('div');
    typingDiv.className = 'message typing';
    typingDiv.id = 'typing-indicator';

    const avatar = state.currentAgent?.name?.[0]?.toUpperCase() || 'A';

    typingDiv.innerHTML = `
        <div class="agent-avatar"><span>${avatar}</span></div>
        <div class="message-content">
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>
    `;

    elements.chatMessages.appendChild(typingDiv);
    scrollToBottom();
}

function hideTypingIndicator() {
    state.isTyping = false;
    handleInputChange();

    const typingIndicator = document.getElementById('typing-indicator');
    if (typingIndicator) {
        typingIndicator.remove();
    }
}

function clearMessages() {
    if (elements.chatMessages) {
        elements.chatMessages.innerHTML = '';
    }
    state.messages = [];
}

function scrollToBottom() {
    if (elements.chatMessages) {
        elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
    }
}

// Agent creation
async function loadCreateView() {
    await populateModelSelector();
    await loadPrebuiltAgents();
    updateCreateButton();
}

async function populateModelSelector() {
    if (!elements.modelSelect) return;

    try {
        const response = await fetch(`${CONFIG.API_BASE}/models/openrouter`);
        if (!response.ok) return;

        const models = await response.json();
        elements.modelSelect.innerHTML = '<option value="">Select a model...</option>';

        Object.entries(models).forEach(([category, modelList]) => {
            const optgroup = document.createElement('optgroup');
            optgroup.label = category;

            modelList.forEach(modelObject => { // Renomeado para clareza
                const option = document.createElement('option');
                option.value = modelObject.name;
                option.textContent = `${modelObject.name} - ${modelObject.cost} credits`;
                optgroup.appendChild(option);
            });

            elements.modelSelect.appendChild(optgroup);
        });
    } catch (error) {
        console.error('Failed to load models:', error);
    }
}

async function loadPrebuiltAgents() {
    if (!elements.prebuiltAgentsGrid) return;

    try {
        const response = await fetch(`${CONFIG.API_BASE}/prebuilt-agents/list`);
        if (response.ok) {
            const agents = await response.json();
            renderPrebuiltAgents(agents);
        }
    } catch (error) {
        console.error('Failed to load prebuilt agents:', error);
    }
}

function renderPrebuiltAgents(agents) {
    elements.prebuiltAgentsGrid.innerHTML = '';

    agents.forEach(agent => {
        const card = document.createElement('div');
        card.className = 'prebuilt-agent-card';
        card.onclick = () => selectPrebuiltAgent(agent);

        card.innerHTML = `
            <div class="agent-avatar"><span>${agent.name[0].toUpperCase()}</span></div>
            <h3>${escapeHtml(agent.name)}</h3>
            <p>${escapeHtml(agent.archetype)}</p>
            <div class="system-badge system-${agent.system_type.toLowerCase()}">${agent.system_type}</div>
        `;

        elements.prebuiltAgentsGrid.appendChild(card);
    });
}

function selectCreationType(type) {
    state.creationMode = type;

    // Update type options
    document.querySelectorAll('.type-option').forEach(option => {
        option.classList.remove('active');
    });

    if (type === 'prebuilt') {
        document.getElementById('prebuilt-option').classList.add('active');
        elements.prebuiltSection.style.display = 'block';
        elements.scratchSection.style.display = 'none';
    } else {
        document.getElementById('scratch-option').classList.add('active');
        elements.prebuiltSection.style.display = 'none';
        elements.scratchSection.style.display = 'block';
    }

    updateCreateButton();
}

function selectSystemType(type) {
    state.selectedSystemType = type;

    // Update system options
    document.querySelectorAll('.system-option').forEach(option => {
        option.classList.remove('active');
    });

    document.getElementById(`${type}-system-option`).classList.add('active');
    updateCreateButton();
}

function selectPrebuiltAgent(agent) {
    state.selectedPrebuiltAgent = agent;

    // Update selection UI
    document.querySelectorAll('.prebuilt-agent-card').forEach(card => {
        card.classList.remove('selected');
    });

    event.currentTarget.classList.add('selected');
    updateCreateButton();
}

function updateCreateButton() {
    if (!elements.createButton) return;

    let canCreate = false;

    if (state.creationMode === 'scratch') {
        const name = document.getElementById('agent-name')?.value?.trim();
        const persona = document.getElementById('agent-persona')?.value?.trim();
        const model = elements.modelSelect?.value;

        canCreate = name && persona && model && state.isAuthenticated;
    } else if (state.creationMode === 'prebuilt') {
        canCreate = state.selectedPrebuiltAgent && state.isAuthenticated;
    }

    elements.createButton.disabled = !canCreate;
}

async function handleAgentCreation(event) {
    event.preventDefault();

    if (!state.isAuthenticated) {
        showError('Please log in to create agents');
        return;
    }

    showLoading(true);

    try {
        let result;

        if (state.creationMode === 'scratch') {
            result = await createScratchAgent();
        } else {
            result = await createPrebuiltAgent();
        }

        if (result) {
            showSuccess(`Agent "${result.name}" created successfully!`);

            // Reload agents list
           await loadMyAgents();
           await loadRecentChats();

            // Reset form
            resetCreateForm();

            // Switch to discover view
            switchView('discover');
        }
    } catch (error) {
        console.error('Agent creation failed:', error);
        showError('Failed to create agent. Please try again.');
    } finally {
        showLoading(false);
    }
}

// WITH THIS FUNCTION
async function createScratchAgent() {
    // We now send JSON instead of FormData to match the Pydantic model
    const agentData = {
        name: document.getElementById('agent-name').value,
        persona: document.getElementById('agent-persona').value,
        detailed_persona: document.getElementById('agent-description').value || '',
        model: elements.modelSelect.value,
        system_type: state.selectedSystemType
        // `is_public` is no longer sent. The backend defaults it to private.
    };

    const response = await fetch(`${CONFIG.API_BASE}/agents/create`, {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${localStorage.getItem('aura_token')}`,
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(agentData)
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || error.error || 'Failed to create agent');
    }

    return await response.json();
}

async function createPrebuiltAgent() {
    const response = await fetch(`${CONFIG.API_BASE}/prebuilt-agents/${state.selectedPrebuiltAgent.id}/create`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${localStorage.getItem('aura_token')}` }
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Failed to create prebuilt agent');
    }

    return await response.json();
}

function resetCreateForm() {
    if (elements.agentForm) {
        elements.agentForm.reset();
    }

    state.selectedPrebuiltAgent = null;
    state.creationMode = 'scratch';
    state.selectedSystemType = 'ceaf';

    // Reset UI
    selectCreationType('scratch');
    selectSystemType('ceaf');

    document.querySelectorAll('.prebuilt-agent-card').forEach(card => {
        card.classList.remove('selected');
    });
}

// Discover content
async function loadDiscoverContent() {
    loadDiscoverCards();
}

function loadDiscoverCards() {
    if (!elements.discoverGrid) return;

    const cards = [
        {
            title: "Creative Writing",
            description: "Explore storytelling and creative expression",
            icon: "✍️"
        },
        {
            title: "Learning Assistant",
            description: "Get help with studies and research",
            icon: "📚"
        },
        {
            title: "Brainstorming",
            description: "Generate ideas and solve problems",
            icon: "💡"
        },
        {
            title: "Role Playing",
            description: "Immersive character interactions",
            icon: "🎭"
        }
    ];

    elements.discoverGrid.innerHTML = '';

    cards.forEach(card => {
        const cardElement = document.createElement('div');
        cardElement.className = 'discover-card';
        cardElement.innerHTML = `
            <div style="font-size: 32px; margin-bottom: 16px;">${card.icon}</div>
            <h3 style="margin-bottom: 8px; font-weight: 600;">${card.title}</h3>
            <p style="color: var(--text-secondary); line-height: 1.4;">${card.description}</p>
        `;

        elements.discoverGrid.appendChild(cardElement);
    });
}

async function loadFeaturedAgents() {
    if (!elements.featuredGrid) return;

    // Create some sample featured agents
    const featured = [
        {
            agent_id: 'sample-1',
            name: 'Creative Writer',
            persona: 'Helps with creative writing and storytelling',
        },
        {
            agent_id: 'sample-2',
            name: 'Study Buddy',
            persona: 'Your personal learning assistant',
        },
        {
            agent_id: 'sample-3',
            name: 'Philosopher',
            persona: 'Explores deep questions about life and existence',
        },
        {
            agent_id: 'sample-4',
            name: 'Coding Mentor',
            persona: 'Helps you learn programming and solve coding problems',
        }
    ];

    elements.featuredGrid.innerHTML = '';

    featured.forEach(agent => {
        const card = document.createElement('div');
        card.className = 'featured-card';
        card.onclick = () => {
            if (state.isAuthenticated) {
                selectAgent(agent);
            } else {
                showAuthModal();
            }
        };

        const avatar = agent.name[0].toUpperCase();

        card.innerHTML = `
            <div class="agent-avatar"><span>${avatar}</span></div>
            <h3>${escapeHtml(agent.name)}</h3>
            <p>${escapeHtml(agent.persona)}</p>
        `;

        elements.featuredGrid.appendChild(card);
    });
}

// Initial data loading
async function loadInitialData() {

    // Load discover content by default
    if (state.isAuthenticated) {
        await loadRecentChats();
    }
}

function initializeUI() {
    // Set initial auth tab
    switchAuthTab('login');

    // Set initial creation mode
    selectCreationType('scratch');
    selectSystemType('ceaf');

    // Initialize input handlers
    handleInputChange();
}

// Event handlers
function handleGlobalKeyDown(event) {
    // ESC to close modals
    if (event.key === 'Escape') {
        if (elements.authModal.style.display === 'flex') {
            hideAuthModal();
        }
        if (elements.userMenu.style.display === 'block') {
            elements.userMenu.style.display = 'none';
        }
        // Add this line to close the agent options modal with ESC
        closeAgentOptionsModal();
    }

    // Ctrl/Cmd + K to focus message input
    if ((event.ctrlKey || event.metaKey) && event.key === 'k' && state.currentView === 'chat') {
        event.preventDefault();
        elements.messageInput?.focus();
    }
}

function handleResize() {
    // Handle responsive behavior
    if (window.innerWidth <= 768) {
        state.sidebarCollapsed = true;
        elements.sidebar?.classList.add('collapsed');
    }
}

function handleGlobalClick(event) {
    // Close user menu when clicking outside
    if (!elements.userProfile.contains(event.target) && !elements.userMenu.contains(event.target)) {
        elements.userMenu.style.display = 'none';
    }
    if (elements.menuBtn && !elements.menuBtn.contains(event.target) && elements.agentDropdownMenu && !elements.agentDropdownMenu.contains(event.target)) {
        elements.agentDropdownMenu.style.display = 'none';
    }

    // ================= START: NEW POPOVER CLOSE LOGIC =================
    // Close mind popovers when clicking outside
    const activePopover = document.querySelector('.mind-popover.visible');
    if (activePopover) {
        // Check if the click was outside the popover AND not on a widget button
        const isWidgetClick = event.target.closest('.mind-widget-btn');
        if (!activePopover.contains(event.target) && !isWidgetClick) {
            closeAllMindPopovers();
        }
    }
    // ================== END: NEW POPOVER CLOSE LOGIC =================
}

// Utility functions
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function showLoading(show) {
    if (elements.loadingOverlay) {
        elements.loadingOverlay.style.display = show ? 'flex' : 'none';
    }
}

function showError(message) {
    if (elements.errorContainer) {
        elements.errorContainer.innerHTML = `<div class="error-message">${escapeHtml(message)}</div>`;
    }
    console.error('Error:', message);
}

function clearError() {
    if (elements.errorContainer) {
        elements.errorContainer.innerHTML = '';
    }
}

function showSuccess(message) {
    // Create a temporary success message
    const successDiv = document.createElement('div');
    successDiv.className = 'success-message';
    successDiv.textContent = message;
    successDiv.style.position = 'fixed';
    successDiv.style.top = '20px';
    successDiv.style.right = '20px';
    successDiv.style.zIndex = '9999';
    successDiv.style.maxWidth = '300px';

    document.body.appendChild(successDiv);

    // Remove after 3 seconds
    setTimeout(() => {
        if (successDiv.parentNode) {
            successDiv.parentNode.removeChild(successDiv);
        }
    }, 3000);

    console.log('Success:', message);
}

// ===============================================
// NEW: Memory Management Functions
// ===============================================
async function showMemoryModal(agentId, agentName) {
    state.currentEditingAgentId = agentId;
    document.getElementById('memory-modal-agent-name').textContent = agentName;
    elements.memoryModal.style.display = 'flex';
    document.body.style.overflow = 'hidden';

    // Reset to the first tab and load its content
    switchMemoryTab('browse');
}

function showAddBiographicalMemoryForm() {
    const form = document.getElementById('add-memory-form');
    form.style.display = 'block';

    // Populate the memory type dropdown based on the current agent's system type
    const agent = state.agents.find(a => a.agent_id === state.currentEditingAgentId);
    const systemType = getSystemType(agent) || 'ncf'; // Default to NCF

    const memoryTypes = {
        ncf: ['Explicit', 'Emotional', 'Procedural', 'Flashbulb', 'Liminal', 'Generative'],
        ceaf: ['failure', 'insight', 'success', 'flashbulb', 'procedural', 'emotional']
    };

    const selectEl = document.getElementById('new-memory-type');
    selectEl.innerHTML = ''; // Clear previous options
    memoryTypes[systemType.toLowerCase()].forEach(type => {
        const option = document.createElement('option');
        option.value = type;
        option.textContent = type;
        selectEl.appendChild(option);
    });

    document.querySelector('.add-memory-btn').style.display = 'none';
}

async function submitNewBiographicalMemory() {
    const content = document.getElementById('new-memory-content').value.trim();
    if (!content) {
        showError("Memory content cannot be empty.");
        return;
    }

    showLoading(true);

    const newMemory = {
        content: content,
        memory_type: document.getElementById('new-memory-type').value,
        emotion_score: parseFloat(document.getElementById('new-memory-emotion').value),
        initial_salience: parseFloat(document.getElementById('new-memory-salience').value),
        custom_metadata: {}
    };

    try {
        const response = await fetch(`${CONFIG.API_BASE}/agents/${state.currentEditingAgentId}/biography/add`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('aura_token')}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ biography: [newMemory] }) // API expects a list
        });

        const result = await response.json();
        if (!response.ok) {
            throw new Error(result.detail || 'Failed to add memory');
        }

        showSuccess(`Successfully added ${result.memories_added} new memory.`);
        hideAddBiographicalMemoryForm();
        // Refresh the memory list to show the new addition
        await loadEditableMemories(state.currentEditingAgentId);

    } catch (error) {
        console.error("Error adding new memory:", error);
        showError(error.message);
    } finally {
        showLoading(false);
    }
}


function hideAddBiographicalMemoryForm() {
    const form = document.getElementById('add-memory-form');
    form.style.display = 'none';
    // Clear the form fields
    document.getElementById('new-memory-content').value = '';
    document.getElementById('new-memory-emotion').value = 0;
    document.getElementById('new-memory-salience').value = 0.5;
    document.querySelector('.add-memory-btn').style.display = 'block';
}

function closeMemoryModal() {
    state.currentEditingAgentId = null;
    state.selectedMemoryFile = null;
    elements.memoryModal.style.display = 'none';
    document.body.style.overflow = 'auto';
}

function switchMemoryTab(tabName) {
    const modal = elements.memoryModal;
    modal.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
    modal.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

    const activeTab = modal.querySelector(`.tab[data-tab="${tabName}"]`);
    const activeContent = document.getElementById(`${tabName}-tab-content`);

    if (activeTab) activeTab.classList.add('active');
    if (activeContent) activeContent.classList.add('active');

    // Load content for the selected tab
    switch(tabName) {
        case 'browse':
            loadEditableMemories(state.currentEditingAgentId);
            break;
        case 'export':
            loadMemoryStatsForExport(state.currentEditingAgentId);
            break;
        case 'analytics':
            document.getElementById('analytics-container').innerHTML = `<p class="placeholder-text">Click "Load Analytics" to view detailed memory insights.</p>`;
            break;
        case 'upload':
             document.getElementById('file-upload-area').innerHTML = `<div class="upload-icon">📁</div><h3>Upload Memory File</h3><p>Click to browse or drag & drop a JSON or CSV file here.</p>`;
             document.getElementById('upload-results-container').innerHTML = '';
             document.getElementById('upload-progress-container').style.display = 'none';
             state.selectedMemoryFile = null;
             break;
    }
}

// Browse & Edit Functions
async function loadEditableMemories(agentId) {
    const container = document.getElementById('memory-list-container');
    container.innerHTML = `<div class="spinner"></div>`;

    try {
        const response = await fetch(`${CONFIG.API_BASE}/agents/${agentId}/memories`, {
            headers: { 'Authorization': `Bearer ${localStorage.getItem('aura_token')}` }
        });
        if (!response.ok) throw new Error('Failed to fetch memories');

        const data = await response.json();
        container.innerHTML = '';

        if (!data.memories || data.memories.length === 0) {
            container.innerHTML = `<p class="placeholder-text">No memories found for this agent.</p>`;
            return;
        }

        data.memories.forEach(mem => {
            const memCard = document.createElement('div');
            memCard.className = 'memory-card';
            memCard.id = `mem-card-${mem.id}`;
            memCard.innerHTML = `
                <div class="memory-card-header">
                    <span class="memory-type-badge">${mem.memory_type}</span>
                    <div class="memory-actions">
                        <button onclick="toggleMemoryEdit('${mem.id}')">Edit</button>
                        <button onclick="deleteMemory('${mem.id}')">Delete</button>
                    </div>
                </div>
                <div class="memory-content-text" id="mem-content-${mem.id}"><p>${escapeHtml(mem.content)}</p></div>
            `;
            container.appendChild(memCard);
        });

    } catch (error) {
        console.error("Error loading memories for editing:", error);
        container.innerHTML = `<p class="placeholder-text" style="color: var(--error);">Could not load memories.</p>`;
    }
}

function toggleMemoryEdit(memId) {
    const contentDiv = document.getElementById(`mem-content-${memId}`);
    const actionsDiv = document.querySelector(`#mem-card-${memId} .memory-actions`);
    const isEditing = contentDiv.querySelector('textarea');

    if (isEditing) { // Cancel edit
        const originalText = isEditing.dataset.original;
        contentDiv.innerHTML = `<p>${escapeHtml(originalText)}</p>`;
        actionsDiv.innerHTML = `<button onclick="toggleMemoryEdit('${memId}')">Edit</button><button onclick="deleteMemory('${memId}')">Delete</button>`;
    } else { // Start edit
        const originalText = contentDiv.textContent;
        contentDiv.innerHTML = `<textarea data-original="${escapeHtml(originalText)}">${originalText}</textarea>`;
        actionsDiv.innerHTML = `<button onclick="saveMemoryEdit('${memId}')">Save</button><button onclick="toggleMemoryEdit('${memId}')">Cancel</button>`;
        contentDiv.querySelector('textarea').focus();
    }
}

async function saveMemoryEdit(memId) {
    const newContent = document.querySelector(`#mem-content-${memId} textarea`).value.trim();
    if (!newContent) {
        showError("Memory content cannot be empty.");
        return;
    }
    showLoading(true);
    try {
        // Step 1: Delete old memory
        const deleteResponse = await fetch(`${CONFIG.API_BASE}/agents/${state.currentEditingAgentId}/memories/${memId}`, {
            method: 'DELETE',
            headers: { 'Authorization': `Bearer ${localStorage.getItem('aura_token')}` }
        });
        if (!deleteResponse.ok) throw new Error('Failed to delete original memory.');

        // Step 2: Add new memory (we need original data to recreate it)
        // This is a limitation. A proper backend 'PUT' would be better.
        // For now, we just create a new 'Explicit' memory.
        const uploadRequest = {
            memories: [{
                content: newContent,
                memory_type: 'Explicit', // Assuming type, this could be improved
            }],
            overwrite_existing: false
        };
        const uploadResponse = await fetch(`${CONFIG.API_BASE}/agents/${state.currentEditingAgentId}/memories/upload`, {
            method: 'POST',
            headers: { 'Authorization': `Bearer ${localStorage.getItem('aura_token')}`, 'Content-Type': 'application/json' },
            body: JSON.stringify(uploadRequest)
        });
        if (!uploadResponse.ok) throw new Error('Failed to save updated memory.');

        showSuccess("Memory updated successfully.");
        await loadEditableMemories(state.currentEditingAgentId); // Refresh list
    } catch (error) {
        console.error("Error saving memory edit:", error);
        showError(error.message);
        await loadEditableMemories(state.currentEditingAgentId); // Refresh to show original state
    } finally {
        showLoading(false);
    }
}

async function deleteMemory(memId) {
    if (!confirm("Are you sure you want to delete this memory permanently?")) return;
    showLoading(true);
    try {
        const response = await fetch(`${CONFIG.API_BASE}/agents/${state.currentEditingAgentId}/memories/${memId}`, {
            method: 'DELETE',
            headers: { 'Authorization': `Bearer ${localStorage.getItem('aura_token')}` }
        });
        if (!response.ok) throw new Error('Failed to delete memory.');
        showSuccess("Memory deleted.");
        document.getElementById(`mem-card-${memId}`).remove();
    } catch (error) {
        console.error("Error deleting memory:", error);
        showError(error.message);
    } finally {
        showLoading(false);
    }
}


// Upload/Export/Analytics Functions
async function loadMemoryStatsForExport(agentId) {
    const container = document.getElementById('export-memory-stats');
    container.innerHTML = `<div class="spinner"></div>`;
    try {
        const response = await fetch(`${CONFIG.API_BASE}/agents/${agentId}/memories/analytics`, {
             headers: { 'Authorization': `Bearer ${localStorage.getItem('aura_token')}` }
        });
        const stats = await response.json();
        container.innerHTML = `
            <div class="stat-card"><div class="stat-number">${stats.total_memories}</div><div class="stat-label">Total Memories</div></div>
            <div class="stat-card"><div class="stat-number">${Object.keys(stats.memory_types).length}</div><div class="stat-label">Memory Types</div></div>
        `;
    } catch (error) {
        container.innerHTML = `<p class="placeholder-text" style="color:var(--error)">Could not load stats.</p>`;
    }
}

async function exportMemories() {
    const format = document.getElementById('export-format-select').value;
    const btn = document.getElementById('export-btn');
    btn.disabled = true;
    btn.innerHTML = `<div class="spinner" style="width:16px;height:16px;border-width:2px;"></div>`;
    try {
        const response = await fetch(`${CONFIG.API_BASE}/agents/${state.currentEditingAgentId}/memories/export?format=${format}`, {
            headers: { 'Authorization': `Bearer ${localStorage.getItem('aura_token')}` }
        });
        if (!response.ok) throw new Error('Export failed');

        const blob = await response.blob();
        const agentName = document.getElementById('memory-modal-agent-name').textContent;
        const filename = `${agentName}_memories.${format}`;

        const a = document.createElement('a');
        a.href = window.URL.createObjectURL(blob);
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(a.href);
        showSuccess("Export successful!");

    } catch (error) {
        console.error("Export error:", error);
        showError("Failed to export memories.");
    } finally {
        btn.disabled = false;
        btn.textContent = "Export Memories";
    }
}

function handleFileDrop(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
        document.getElementById('memory-file-input').files = e.dataTransfer.files;
        handleFileSelect({ target: { files: e.dataTransfer.files } });
    }
}

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (!file) return;
    state.selectedMemoryFile = file;
    document.getElementById('file-upload-area').innerHTML = `<div class="upload-icon">📄</div><h3>${file.name}</h3><p>${(file.size / 1024).toFixed(1)} KB</p>`;
    // Trigger upload automatically
    uploadMemories();
}

async function uploadMemories() {
    if (!state.selectedMemoryFile) {
        showError("Please select a file to upload.");
        return;
    }
    const progressContainer = document.getElementById('upload-progress-container');
    const progressFill = document.getElementById('upload-progress-fill');
    const resultsContainer = document.getElementById('upload-results-container');

    progressContainer.style.display = 'block';
    progressFill.style.width = '0%';
    resultsContainer.innerHTML = '';

    try {
        const formData = new FormData();
        formData.append('file', state.selectedMemoryFile);

        // Simulate progress
        setTimeout(() => progressFill.style.width = '50%', 100);

        const response = await fetch(`${CONFIG.API_BASE}/agents/${state.currentEditingAgentId}/memories/upload/file`, {
            method: 'POST',
            headers: { 'Authorization': `Bearer ${localStorage.getItem('aura_token')}` },
            body: formData
        });

        progressFill.style.width = '100%';
        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.detail || 'Upload failed');
        }

        resultsContainer.innerHTML = `<div class="success-message"><strong>Upload Complete!</strong><br>Successful: ${result.successful_uploads}<br>Failed: ${result.failed_uploads}</div>`;
        if (result.errors.length > 0) {
            resultsContainer.innerHTML += `<div style="font-size:12px;color:var(--text-secondary);margin-top:8px;">Errors: ${result.errors.join(', ')}</div>`;
        }
        showSuccess("Memories uploaded.");
    } catch (error) {
        resultsContainer.innerHTML = `<div class="error-message">${error.message}</div>`;
    }
}

async function loadAnalytics() {
    const container = document.getElementById('analytics-container');
    const btn = document.getElementById('analytics-btn');
    btn.disabled = true;
    container.innerHTML = `<div class="spinner"></div>`;

    try {
        const response = await fetch(`${CONFIG.API_BASE}/agents/${state.currentEditingAgentId}/memories/analytics`, {
            headers: { 'Authorization': `Bearer ${localStorage.getItem('aura_token')}` }
        });
        if (!response.ok) throw new Error('Failed to load analytics');
        const stats = await response.json();

        container.innerHTML = `
            <div class="memory-stats-grid">
                <div class="stat-card"><div class="stat-number">${stats.total_memories}</div><div class="stat-label">Total Memories</div></div>
                <div class="stat-card"><div class="stat-number">${Object.keys(stats.memory_types).length}</div><div class="stat-label">Memory Types</div></div>
                <div class="stat-card"><div class="stat-number">${stats.recent_activity.last_7_days}</div><div class="stat-label">Added Last 7 Days</div></div>
            </div>
            <div class="analytics-grid">
                <div class="analytics-card">
                    <h4>Memory Type Distribution</h4>
                    ${Object.entries(stats.memory_types).map(([type, count]) => `<div class="analytics-item"><span>${type}</span><span>${count}</span></div>`).join('')}
                </div>
                <div class="analytics-card">
                    <h4>Emotion Distribution</h4>
                    <div class="analytics-item"><span>Positive</span><span style="color:var(--success)">${stats.emotion_distribution.positive}</span></div>
                    <div class="analytics-item"><span>Neutral</span><span style="color:var(--text-secondary)">${stats.emotion_distribution.neutral}</span></div>
                    <div class="analytics-item"><span>Negative</span><span style="color:var(--error)">${stats.emotion_distribution.negative}</span></div>
                </div>
            </div>
        `;

    } catch (error) {
        container.innerHTML = `<p class="placeholder-text" style="color:var(--error)">${error.message}</p>`;
    } finally {
        btn.disabled = false;
    }
}

// Make functions globally available for onclick handlers
window.switchAuthTab = switchAuthTab;
window.selectCreationType = selectCreationType;
window.selectSystemType = selectSystemType;

window.showAddBiographicalMemoryForm = showAddBiographicalMemoryForm;
window.hideAddBiographicalMemoryForm = hideAddBiographicalMemoryForm;
window.submitNewBiographicalMemory = submitNewBiographicalMemory;

window.startChatWithAgent = startChatWithAgent;
window.cloneAgentToLibrary = cloneAgentToLibrary;
window.closeAgentOptionsModal = closeAgentOptionsModal;
window.handleAuth = handleAuth;
window.selectAgent = selectAgent;
window.showMemoryModal = showMemoryModal;
window.closeMemoryModal = closeMemoryModal;
window.toggleMemoryEdit = toggleMemoryEdit;
window.saveMemoryEdit = saveMemoryEdit;
window.deleteMemory = deleteMemory;

window.showProfileModal = showProfileModal;
window.closeProfileModal = closeProfileModal;
window.showFilesModal = showFilesModal;
window.closeFilesModal = closeFilesModal;

window.loadMyAgents = loadMyAgents;
window.deleteAgent = deleteAgent;
window.unpublishAgent = unpublishAgent;
window.showSetPriceModal = showSetPriceModal;
window.closeSetPriceModal = closeSetPriceModal;
window.saveAgentPrice = saveAgentPrice;

