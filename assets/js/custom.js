// Put your custom JS code here

const getTheme = () => document.querySelector('html').getAttribute('data-bs-theme');

const updateLogos = () => {
    const theme = getTheme();
    const logos = {
        hf: {
            light: 'images/hf-logo-light.jpg',
            dark: 'images/hf-logo-dark.png'
        },
        entalpic: {
            light: 'images/entalpic-logo-light.png',
            dark: 'images/entalpic-logo-dark.png'
        }
    };
    document.getElementById('hf-logo').src = logos.hf[theme];
    document.getElementById('entalpic-logo').src = logos.entalpic[theme];
};

function ready(fn) {
    if (document.readyState !== 'loading') {
        fn();
        return;
    }
    document.addEventListener('DOMContentLoaded', fn);
}
ready(function () {
    // updateLogos();
});

