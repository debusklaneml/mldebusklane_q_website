/**
 * Theme Toggle Implementation
 * Manages dark/light mode switching and persistence
 */

(function() {
  'use strict';

  // Check for saved theme preference or default to light mode
  const currentTheme = localStorage.getItem('theme') || 'light';

  // Apply theme on page load (before page renders to prevent flash)
  if (currentTheme === 'dark') {
    document.documentElement.classList.add('dark-mode');
    document.body.classList.add('dark-mode');
  }

  // Create and inject theme toggle button
  function createThemeToggle() {
    const toggle = document.createElement('button');
    toggle.className = 'theme-toggle';
    toggle.setAttribute('aria-label', 'Toggle theme');
    toggle.innerHTML = currentTheme === 'dark' ? '☀️' : '🌙';

    // Insert into header or body
    const header = document.getElementById('quarto-header');
    if (header) {
      header.style.position = 'relative';
      header.appendChild(toggle);
    } else {
      // Fallback: append to body
      document.body.appendChild(toggle);
    }

    return toggle;
  }

  // Toggle theme function
  function toggleTheme() {
    const html = document.documentElement;
    const body = document.body;
    const isDark = body.classList.toggle('dark-mode');
    html.classList.toggle('dark-mode');

    const theme = isDark ? 'dark' : 'light';

    // Save preference
    localStorage.setItem('theme', theme);

    // Update button icon
    const toggle = document.querySelector('.theme-toggle');
    if (toggle) {
      toggle.innerHTML = isDark ? '☀️' : '🌙';
    }

    // If using Giscus comments, sync theme
    syncGiscusTheme(theme);
  }

  // Sync with Giscus comments (if implemented later)
  function syncGiscusTheme(theme) {
    const giscusFrame = document.querySelector('iframe.giscus-frame');
    if (giscusFrame) {
      const giscusTheme = theme === 'dark' ? 'dark' : 'light';
      giscusFrame.contentWindow.postMessage(
        { giscus: { setConfig: { theme: giscusTheme } } },
        'https://giscus.app'
      );
    }
  }

  // Initialize on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  function init() {
    const toggle = createThemeToggle();
    toggle.addEventListener('click', toggleTheme);
  }
})();
