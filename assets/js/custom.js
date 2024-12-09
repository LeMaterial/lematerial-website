// Put your custom JS code here

const getTheme = () => document.querySelector('html').getAttribute('data-bs-theme');

const updateLogos = () => {
    const theme = getTheme();
    const logos = {
        hf: {
            light: 'https://github.com/LeMaterial/lematerial-website/blob/main/images/assets/images/hf-logo-light.jpg?raw=true',
            dark: 'https://github.com/LeMaterial/lematerial-website/blob/main/images/assets/images/hf-logo-dark.png?raw=true'
        },
        entalpic: {
            light: 'https://github.com/LeMaterial/lematerial-website/blob/main/images/assets/images/entalpic.png?raw=true',
            dark: 'https://github.com/LeMaterial/lematerial-website/blob/main/images/assets/images/entalpic.png?raw=true'
        }
    };
    const hfLogo = document.getElementById('hf-logo');
    if (hfLogo) hfLogo.src = logos.hf[theme];

    const entalpicLogo = document.getElementById('entalpic-logo');
    if (entalpicLogo) entalpicLogo.src = logos.entalpic[theme];
};

function ready(fn) {
    if (document.readyState !== 'loading') {
        fn();
        return;
    }
    document.addEventListener('DOMContentLoaded', fn);
}
ready(function () {
    updateLogos();
    document.getElementById('buttonColorMode').addEventListener('click', updateLogos);
});

