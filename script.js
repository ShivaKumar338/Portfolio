// script.js - Interactive behaviors, Typewriter, Terminal, Projects Filter & Modal

document.addEventListener('DOMContentLoaded', () => {
  initTypewriter();
  initHeaderScroll();
  initMobileDrawer();
  initStatsCounter();
  initProjectsGrid();
  initModal();
  initTerminal();
  initCopyButtons();
});

/* ==========================================================================
   1. Typewriter Animation
   ========================================================================== */
function initTypewriter() {
  const el = document.getElementById('typewriter');
  if (!el) return;

  const roles = [
    "AI & Machine Learning Engineer",
    "Autonomous Agentic AI Developer",
    "Computer Vision & Edge AI Specialist",
    "FastAPI & Full-Stack Platform Builder"
  ];

  let roleIdx = 0;
  let charIdx = 0;
  let isDeleting = false;
  const typeSpeed = 70;
  const deleteSpeed = 35;
  const pauseTime = 1800;

  function type() {
    const currentRole = roles[roleIdx];

    if (isDeleting) {
      el.textContent = currentRole.substring(0, charIdx - 1);
      charIdx--;
    } else {
      el.textContent = currentRole.substring(0, charIdx + 1);
      charIdx++;
    }

    let nextDelay = isDeleting ? deleteSpeed : typeSpeed;

    if (!isDeleting && charIdx === currentRole.length) {
      nextDelay = pauseTime;
      isDeleting = true;
    } else if (isDeleting && charIdx === 0) {
      isDeleting = false;
      roleIdx = (roleIdx + 1) % roles.length;
      nextDelay = 400;
    }

    setTimeout(type, nextDelay);
  }

  type();
}

/* ==========================================================================
   2. Header Scroll & Active Scrollspy
   ========================================================================== */
function initHeaderScroll() {
  const header = document.getElementById('header');
  const sections = document.querySelectorAll('section[id]');
  const navLinks = document.querySelectorAll('.desktop-nav .nav-link');

  window.addEventListener('scroll', () => {
    if (window.scrollY > 40) {
      header.classList.add('scrolled');
    } else {
      header.classList.remove('scrolled');
    }

    // Scrollspy
    let currentId = '';
    sections.forEach(section => {
      const sectionTop = section.offsetTop - 120;
      const sectionHeight = section.offsetHeight;
      if (window.scrollY >= sectionTop && window.scrollY < sectionTop + sectionHeight) {
        currentId = section.getAttribute('id');
      }
    });

    navLinks.forEach(link => {
      link.classList.remove('active');
      if (link.getAttribute('href') === `#${currentId}`) {
        link.classList.add('active');
      }
    });
  });
}

/* ==========================================================================
   3. Mobile Navigation Drawer
   ========================================================================== */
function initMobileDrawer() {
  const toggleBtn = document.getElementById('mobile-toggle');
  const drawer = document.getElementById('mobile-drawer');
  const links = drawer.querySelectorAll('.mobile-link');

  if (!toggleBtn || !drawer) return;

  toggleBtn.addEventListener('click', () => {
    drawer.classList.toggle('open');
  });

  links.forEach(link => {
    link.addEventListener('click', () => {
      drawer.classList.remove('open');
    });
  });
}

/* ==========================================================================
   4. Stats Number Counter Animation
   ========================================================================== */
function initStatsCounter() {
  const statNumbers = document.querySelectorAll('.stat-number');
  if (!statNumbers.length) return;

  let animated = false;

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting && !animated) {
        animated = true;
        statNumbers.forEach(num => {
          const target = parseInt(num.getAttribute('data-target'), 10);
          const hasPlus = target === 20;
          const hasPct = target === 15;
          let current = 0;
          const step = Math.max(1, Math.floor(target / 40));
          const interval = setInterval(() => {
            current += step;
            if (current >= target) {
              current = target;
              clearInterval(interval);
              num.textContent = current + (hasPlus ? '+' : '') + (hasPct ? '%' : '');
            } else {
              num.textContent = current + (hasPlus ? '+' : '') + (hasPct ? '%' : '');
            }
          }, 35);
        });
      }
    });
  }, { threshold: 0.5 });

  const statsStrip = document.getElementById('stats-strip');
  if (statsStrip) observer.observe(statsStrip);
}

/* ==========================================================================
   5. Dynamic Projects Grid, Filter & Search
   ========================================================================== */
let currentCategory = 'all';
let currentSearchQuery = '';

function initProjectsGrid() {
  const grid = document.getElementById('projects-grid');
  if (!grid || typeof PROJECTS_DATA === 'undefined') return;

  updateFilterCounts();
  renderProjects();

  // Category filter tabs
  const filterBtns = document.querySelectorAll('.filter-btn');
  filterBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      filterBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentCategory = btn.getAttribute('data-filter');
      renderProjects();
    });
  });

  // Search input
  const searchInput = document.getElementById('project-search');
  if (searchInput) {
    searchInput.addEventListener('input', (e) => {
      currentSearchQuery = e.target.value.toLowerCase().trim();
      renderProjects();
    });
  }
}

function updateFilterCounts() {
  const counts = {
    all: PROJECTS_DATA.length,
    agentic: PROJECTS_DATA.filter(p => p.category === 'agentic').length,
    vision: PROJECTS_DATA.filter(p => p.category === 'vision').length,
    ml: PROJECTS_DATA.filter(p => p.category === 'ml').length,
    fullstack: PROJECTS_DATA.filter(p => p.category === 'fullstack').length,
  };

  for (const [key, val] of Object.entries(counts)) {
    const el = document.getElementById(`count-${key}`);
    if (el) el.textContent = val;
  }
}

function renderProjects() {
  const grid = document.getElementById('projects-grid');
  if (!grid) return;

  const filtered = PROJECTS_DATA.filter(project => {
    // Category check
    const matchesCategory = currentCategory === 'all' || project.category === currentCategory;

    // Search check
    const matchesSearch = !currentSearchQuery || 
      project.title.toLowerCase().includes(currentSearchQuery) ||
      project.summary.toLowerCase().includes(currentSearchQuery) ||
      project.tagline.toLowerCase().includes(currentSearchQuery) ||
      project.stack.some(s => s.toLowerCase().includes(currentSearchQuery));

    return matchesCategory && matchesSearch;
  });

  if (filtered.length === 0) {
    grid.innerHTML = `
      <div style="grid-column: 1/-1; text-align: center; padding: 60px 20px; color: var(--text-muted);">
        <p style="font-size: 18px; margin-bottom: 8px;">No projects found matching "${currentSearchQuery}"</p>
        <p style="font-size: 14px;">Try searching for "FastAPI", "Gemini", "YOLO", or select another category filter.</p>
      </div>
    `;
    return;
  }

  grid.innerHTML = filtered.map(p => `
    <article class="project-card glass-card" id="card-${p.id}">
      <div class="project-top-row">
        <span class="project-category-badge">${p.badge}</span>
        ${p.liveUrl ? `
          <a href="${p.liveUrl}" target="_blank" rel="noopener" class="project-live-indicator" title="Live Deployment">
            <span class="status-dot" style="background-color: var(--accent-emerald);"></span>
            <span>Live Demo</span>
          </a>
        ` : ''}
      </div>

      <h3 class="project-title">${p.title}</h3>
      <div class="project-tagline">${p.tagline}</div>
      <p class="project-summary">${p.summary}</p>

      <div class="project-metric-box">
        <strong>Impact:</strong> ${p.metrics}
      </div>

      <div class="project-stack">
        ${p.stack.slice(0, 5).map(tech => `<span class="stack-pill">${tech}</span>`).join('')}
        ${p.stack.length > 5 ? `<span class="stack-pill">+${p.stack.length - 5}</span>` : ''}
      </div>

      <div class="project-actions">
        <button class="btn btn-outline btn-card-primary" onclick="openProjectModal('${p.id}')">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>
          <span>Architecture & Details</span>
        </button>
        <a href="${p.githubUrl}" target="_blank" rel="noopener" class="btn btn-ghost btn-card-link" title="Open GitHub Repo">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"></path></svg>
          <span>GitHub</span>
        </a>
      </div>
    </article>
  `).join('');
}

/* ==========================================================================
   6. Project Modal Logic
   ========================================================================== */
function initModal() {
  const modal = document.getElementById('project-modal');
  const closeBtn = document.getElementById('modal-close-btn');
  const backdrop = document.getElementById('modal-backdrop');

  if (!modal) return;

  const closeModal = () => {
    modal.classList.remove('active');
    modal.setAttribute('aria-hidden', 'true');
    document.body.style.overflow = '';
  };

  closeBtn.addEventListener('click', closeModal);
  backdrop.addEventListener('click', closeModal);

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && modal.classList.contains('active')) {
      closeModal();
    }
  });
}

window.openProjectModal = function(projectId) {
  const modal = document.getElementById('project-modal');
  const content = document.getElementById('modal-content');
  if (!modal || !content || typeof PROJECTS_DATA === 'undefined') return;

  const project = PROJECTS_DATA.find(p => p.id === projectId);
  if (!project) return;

  content.innerHTML = `
    <div class="modal-badge-row">
      <span class="project-category-badge">${project.badge}</span>
      <span class="mini-tag" style="color: var(--primary-cyan);">${project.category.toUpperCase()}</span>
      ${project.liveUrl ? `
        <a href="${project.liveUrl}" target="_blank" rel="noopener" class="project-live-indicator">
          <span class="status-dot" style="background-color: var(--accent-emerald);"></span>
          <span>Live Demo Available</span>
        </a>
      ` : ''}
    </div>

    <h2 class="modal-title">${project.title}</h2>
    <div class="modal-tagline">${project.tagline}</div>

    <div class="modal-summary-box">
      <strong>Core Overview:</strong><br>
      ${project.summary}
    </div>

    <div class="project-metric-box">
      <strong>Key Performance Metric / Milestone:</strong><br>
      ${project.metrics}
    </div>

    <h4 class="modal-section-title">Key Architectural Features & Engineering Highlights</h4>
    <ul class="modal-highlights-list">
      ${project.highlights.map(h => `<li>${h}</li>`).join('')}
    </ul>

    <h4 class="modal-section-title">System Dataflow Architecture</h4>
    <div class="modal-arch-box">
      ${project.architecture}
    </div>

    <h4 class="modal-section-title">Full Technology Stack</h4>
    <div class="project-stack" style="margin-bottom: 24px;">
      ${project.stack.map(s => `<span class="stack-pill" style="font-size: 12px; padding: 5px 12px;">${s}</span>`).join('')}
    </div>

    <div class="modal-footer-actions">
      <a href="${project.githubUrl}" target="_blank" rel="noopener" class="btn btn-primary" style="flex: 1;">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"></path></svg>
        <span>Explore GitHub Repository</span>
      </a>
      ${project.liveUrl ? `
        <a href="${project.liveUrl}" target="_blank" rel="noopener" class="btn btn-outline" style="flex: 1;">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path><polyline points="15 3 21 3 21 9"></polyline><line x1="10" y1="14" x2="21" y2="3"></line></svg>
          <span>Open Live Deployment</span>
        </a>
      ` : ''}
    </div>
  `;

  modal.classList.add('active');
  modal.setAttribute('aria-hidden', 'false');
  document.body.style.overflow = 'hidden';
};

/* ==========================================================================
   7. Interactive Recruiter Terminal
   ========================================================================== */
function initTerminal() {
  const terminalForm = document.getElementById('terminal-form');
  const terminalInput = document.getElementById('terminal-input');
  const terminalOutput = document.getElementById('terminal-output');
  const quickBtns = document.querySelectorAll('.term-quick-btn');

  if (!terminalForm || !terminalInput || !terminalOutput) return;

  const commands = {
    help: () => `
      <span class="term-highlight">Available Commands:</span><br>
      &nbsp;&nbsp;<strong>skills</strong>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Technical stack and proficiency levels<br>
      &nbsp;&nbsp;<strong>projects</strong>&nbsp;&nbsp;&nbsp;- Top production builds with links<br>
      &nbsp;&nbsp;<strong>exp</strong>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Industry internship history (EdiGlobe, SkillDuniya)<br>
      &nbsp;&nbsp;<strong>education</strong>&nbsp;&nbsp;- Degree, institution & coursework<br>
      &nbsp;&nbsp;<strong>api</strong>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Live FastAPI backend status and Swagger docs<br>
      &nbsp;&nbsp;<strong>contact</strong>&nbsp;&nbsp;&nbsp;&nbsp;- Direct email, LinkedIn and GitHub handles<br>
      &nbsp;&nbsp;<strong>hire</strong>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Ready-to-send internship interview message<br>
      &nbsp;&nbsp;<strong>clear</strong>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Clear the terminal screen<br>
    `,

    api: () => `
      <span class="term-highlight">FastAPI Serverless Backend (Vercel):</span><br>
      &nbsp;&nbsp;Health Check:&nbsp;&nbsp;&nbsp;&nbsp;<a href="/api/health" target="_blank" class="term-highlight">/api/health</a><br>
      &nbsp;&nbsp;Projects API:&nbsp;&nbsp;&nbsp;&nbsp;<a href="/api/projects" target="_blank" class="term-highlight">/api/projects</a><br>
      &nbsp;&nbsp;Interactive Docs:&nbsp;<a href="/api/docs" target="_blank" class="term-highlight">/api/docs (Swagger UI)</a><br>
      &nbsp;&nbsp;Runtime:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Python 3.12 ASGI on Vercel Serverless
    `,

    skills: () => `
      <span class="term-highlight">Core Technical Competencies:</span><br>
      &nbsp;&nbsp;[AI & Deep Learning]&nbsp;&nbsp;&nbsp;PyTorch, TensorFlow, YOLOv8, OpenCV, TrOCR, Scikit-learn<br>
      &nbsp;&nbsp;[Agentic AI & LLMs]&nbsp;&nbsp;&nbsp;&nbsp;CrewAI, Gemini 1.5, RAG Architectures, Multi-Agent Flow<br>
      &nbsp;&nbsp;[Backend Engineering]&nbsp;&nbsp;FastAPI, Python, Flask, PostgreSQL, Supabase, Celery, Redis<br>
      &nbsp;&nbsp;[Frontend & Web]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;JavaScript (ES6+), React, Next.js 16, React Flow, HTML/CSS<br>
      &nbsp;&nbsp;[Algorithms & DSA]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;100+ NeetCode/LeetCode problems solved in Python
    `,

    projects: () => `
      <span class="term-highlight">Featured Repositories (github.com/ShivaKumar338):</span><br>
      &nbsp;&nbsp;1. <strong>OrchestAI</strong>: Full-stack workforce orchestration with DAGs (FastAPI + Next.js 16)<br>
      &nbsp;&nbsp;2. <strong>DeepGuard AI</strong>: Multimodal deepfake detection agent across 4 modalities<br>
      &nbsp;&nbsp;3. <strong>AI Technician Routing</strong>: Geospatial dispatch + twin-agent Gemini automated negotiation<br>
      &nbsp;&nbsp;4. <strong>VisionAI</strong>: Real-time YOLOv8 + TrOCR voice & vision edge assistant<br>
      &nbsp;&nbsp;5. <strong>CrewAI Pipeline</strong>: 3-agent autonomous market intelligence Flow<br>
      &nbsp;&nbsp;6. <strong>Fraud Detection</strong>: Live ML model deployed on Render (fraud-detection-dnty.onrender.com)<br>
      <em>Tip: Click any project card above for interactive architectural breakdown!</em>
    `,

    exp: () => `
      <span class="term-highlight">Internship Experience:</span><br>
      &nbsp;&nbsp;• <strong>EdiGlobe (IIT Kharagpur Initiative)</strong> | Oct 2025 – Dec 2025<br>
      &nbsp;&nbsp;&nbsp;&nbsp;Role: Internship Trainee — Neural Networks & Supervised ML<br>
      &nbsp;&nbsp;&nbsp;&nbsp;Trained PyTorch/TensorFlow models, designed automated preprocessing pipelines.<br><br>
      &nbsp;&nbsp;• <strong>SkillDuniya</strong> | Mar 2025 – May 2025<br>
      &nbsp;&nbsp;&nbsp;&nbsp;Role: Training AI Intern<br>
      &nbsp;&nbsp;&nbsp;&nbsp;Built predictive models with Scikit-Learn, Pandas, and automated analytics routines.
    `,

    education: () => `
      <span class="term-highlight">Academic Background:</span><br>
      &nbsp;&nbsp;<strong>B.Tech in Artificial Intelligence and Machine Learning</strong> (2024 - 2028)<br>
      &nbsp;&nbsp;Pallavi Engineering College — Greater Hyderabad Area, India<br>
      &nbsp;&nbsp;Focus: Machine Learning Systems, Neural Networks, Database Design, DSA.
    `,

    contact: () => `
      <span class="term-highlight">Direct Contact Channels:</span><br>
      &nbsp;&nbsp;Phone / Call:&nbsp;&nbsp;<a href="tel:+918520929979" class="term-highlight">+91 8520929979</a><br>
      &nbsp;&nbsp;WhatsApp:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://wa.me/918520929979" target="_blank" class="term-highlight">+91 8520929979</a><br>
      &nbsp;&nbsp;Email:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="mailto:mshivakumar1289@gmail.com" class="term-highlight">mshivakumar1289@gmail.com</a><br>
      &nbsp;&nbsp;LinkedIn:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://www.linkedin.com/in/shiva-kumar-madagoni-627038327/" target="_blank" class="term-highlight">shiva-kumar-madagoni-627038327</a><br>
      &nbsp;&nbsp;GitHub:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://github.com/ShivaKumar338" target="_blank" class="term-highlight">github.com/ShivaKumar338</a><br>
      &nbsp;&nbsp;LeetCode:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://leetcode.com/u/shivakumar338/" target="_blank" class="term-highlight">leetcode.com/u/shivakumar338</a>
    `,

    hire: () => {
      showToast("Triggering email client for interview request...");
      setTimeout(() => {
        window.location.href = "mailto:mshivakumar1289@gmail.com?subject=Internship%20Opportunity%20-%20Interview%20Invitation&body=Hello%20Shiva,%0D%0A%0D%0AWe%20reviewed%20your%20portfolio%20and%20AI/ML%20projects.%20We%20would%20love%20to%20discuss%20an%20internship%20opportunity%20with%20our%20engineering%20team.%0D%0A%0D%0ABest%20regards,";
      }, 800);
      return `
        <span class="term-highlight" style="color: #34d399;">Direct Contact for Opportunities:</span><br>
        📞 Phone: <a href="tel:+918520929979" style="color: var(--primary-cyan);">+91 8520929979</a><br>
        💬 WhatsApp: <a href="https://wa.me/918520929979" target="_blank" style="color: var(--accent-emerald);">+91 8520929979</a><br>
        📧 Email: <a href="mailto:mshivakumar1289@gmail.com" style="color: var(--primary-cyan);">mshivakumar1289@gmail.com</a>
      `;
    },

    clear: () => {
      terminalOutput.innerHTML = '';
      return null;
    }
  };

  function executeCommand(rawCmd) {
    const cleanCmd = rawCmd.trim().toLowerCase();
    if (!cleanCmd) return;

    // Echo input
    const promptLine = document.createElement('div');
    promptLine.className = 'term-line term-prompt-line';
    promptLine.innerHTML = `<span style="color: var(--accent-emerald);">shiva@internship:~$</span> ${cleanCmd}`;
    terminalOutput.appendChild(promptLine);

    if (cleanCmd === 'clear') {
      commands.clear();
      return;
    }

    const handler = commands[cleanCmd];
    const responseLine = document.createElement('div');
    responseLine.className = 'term-line term-response';

    if (handler) {
      responseLine.innerHTML = handler();
    } else {
      responseLine.innerHTML = `command not found: <em>${cleanCmd}</em>. Type <span class="term-highlight">help</span> for valid commands.`;
    }

    terminalOutput.appendChild(responseLine);
    terminalOutput.scrollTop = terminalOutput.scrollHeight;
  }

  terminalForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const val = terminalInput.value;
    terminalInput.value = '';
    executeCommand(val);
  });

  quickBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const cmd = btn.getAttribute('data-cmd');
      executeCommand(cmd);
    });
  });
}

/* ==========================================================================
   8. Copy Phone & Email to Clipboard & Toast Notifications
   ========================================================================== */
function initCopyButtons() {
  document.querySelectorAll('.copy-email-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      const email = btn.getAttribute('data-email') || "mshivakumar1289@gmail.com";
      navigator.clipboard.writeText(email).then(() => {
        showToast(`Copied ${email} to clipboard!`);
      }).catch(() => {
        showToast(`Email: ${email}`);
      });
    });
  });

  document.querySelectorAll('.copy-phone-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      const phone = btn.getAttribute('data-phone') || "+91 8520929979";
      navigator.clipboard.writeText(phone).then(() => {
        showToast(`Copied ${phone} to clipboard!`);
      }).catch(() => {
        showToast(`Phone: ${phone}`);
      });
    });
  });
}

function showToast(message) {
  const container = document.getElementById('toast-container');
  if (!container) return;

  const toast = document.createElement('div');
  toast.className = 'toast';
  toast.innerHTML = `
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>
    <span>${message}</span>
  `;

  container.appendChild(toast);

  setTimeout(() => {
    toast.remove();
  }, 3000);
}
