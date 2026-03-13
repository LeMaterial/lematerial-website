function ready(fn) {
    if (document.readyState !== 'loading') {
        fn();
        return;
    }
    document.addEventListener('DOMContentLoaded', fn);
}
ready(function () {
    const announcement = document.getElementById('announcement');
    const announcementClose = announcement ? announcement.querySelector('.btn-close') : null;
    if (announcement && announcementClose) {
        announcementClose.addEventListener('click', function () {
            // Fallback for setups without Bootstrap alert JS: close and persist dismissal.
            if (announcement.parentNode) {
                announcement.remove();
            }

            const alertId = announcement.dataset.id;
            if (alertId) {
                try {
                    localStorage.setItem(alertId, 'closed');
                } catch (error) {
                    // Ignore storage errors (e.g. private mode restrictions).
                }
            }
            document.documentElement.setAttribute('data-global-alert', 'closed');
        });
    }

    const updatesAnchor = document.getElementById('latest-updates');
    const alertLink = document.querySelector('a.alert-link[href*="#latest-updates"]');
    if (updatesAnchor && alertLink) {
        alertLink.addEventListener('click', function (event) {
            event.preventDefault();
            updatesAnchor.scrollIntoView({ behavior: 'smooth', block: 'start' });
            if (window.history && window.history.replaceState) {
                window.history.replaceState({}, document.title, window.location.pathname + '#latest-updates');
            } else {
                window.location.hash = 'latest-updates';
            }
        });
    }

    const embedTriggers = document.querySelectorAll('[data-rg-load-embed]');
    embedTriggers.forEach(function (trigger) {
        trigger.addEventListener('click', function () {
            const embedContainer = trigger.closest('[data-rg-embed]');
            if (!embedContainer) {
                return;
            }
            const iframe = embedContainer.parentElement.querySelector('iframe[data-src]');
            if (!iframe) {
                return;
            }
            if (!iframe.getAttribute('src')) {
                iframe.setAttribute('src', iframe.getAttribute('data-src'));
            }
            iframe.hidden = false;
            embedContainer.remove();
        });
    });

    const offcanvasMain = document.getElementById('offcanvasNavMain');
    const mobileDropdownToggles = offcanvasMain
        ? offcanvasMain.querySelectorAll('.nav-item.dropdown > .dropdown-toggle')
        : [];
    const mobileBreakpoint = window.matchMedia('(max-width: 991.98px)');

    function closeMobileDropdowns() {
        mobileDropdownToggles.forEach(function (otherToggle) {
            const otherMenu = otherToggle.nextElementSibling;
            if (otherMenu && otherMenu.classList.contains('dropdown-menu')) {
                otherMenu.classList.remove('show');
            }
            otherToggle.setAttribute('aria-expanded', 'false');
        });
    }

    mobileDropdownToggles.forEach(function (toggle) {
        toggle.addEventListener('click', function (event) {
            if (!mobileBreakpoint.matches) {
                return;
            }

            const menu = toggle.nextElementSibling;
            if (!menu || !menu.classList.contains('dropdown-menu')) {
                return;
            }

            event.preventDefault();
            const isOpen = menu.classList.contains('show');
            closeMobileDropdowns();

            if (!isOpen) {
                menu.classList.add('show');
                toggle.setAttribute('aria-expanded', 'true');
            }
        });
    });
});
