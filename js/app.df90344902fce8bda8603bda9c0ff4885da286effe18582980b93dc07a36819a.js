"use strict";
(() => {
  var __create = Object.create;
  var __defProp = Object.defineProperty;
  var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
  var __getOwnPropNames = Object.getOwnPropertyNames;
  var __getProtoOf = Object.getPrototypeOf;
  var __hasOwnProp = Object.prototype.hasOwnProperty;
  var __commonJS = (cb, mod) => function __require() {
    return mod || (0, cb[__getOwnPropNames(cb)[0]])((mod = { exports: {} }).exports, mod), mod.exports;
  };
  var __copyProps = (to, from, except, desc) => {
    if (from && typeof from === "object" || typeof from === "function") {
      for (let key of __getOwnPropNames(from))
        if (!__hasOwnProp.call(to, key) && key !== except)
          __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
    }
    return to;
  };
  var __toESM = (mod, isNodeMode, target) => (target = mod != null ? __create(__getProtoOf(mod)) : {}, __copyProps(
    // If the importer is in node compatibility mode or this is not an ESM
    // file that has been converted to a CommonJS file using a Babel-
    // compatible transform (i.e. "__esModule" has not been set), then set
    // "default" to the CommonJS "module.exports" for node compatibility.
    isNodeMode || !mod || !mod.__esModule ? __defProp(target, "default", { value: mod, enumerable: true }) : target,
    mod
  ));
  var __async = (__this, __arguments, generator) => {
    return new Promise((resolve, reject) => {
      var fulfilled = (value) => {
        try {
          step(generator.next(value));
        } catch (e2) {
          reject(e2);
        }
      };
      var rejected = (value) => {
        try {
          step(generator.throw(value));
        } catch (e2) {
          reject(e2);
        }
      };
      var step = (x) => x.done ? resolve(x.value) : Promise.resolve(x.value).then(fulfilled, rejected);
      step((generator = generator.apply(__this, __arguments)).next());
    });
  };

  // node_modules/lazysizes/lazysizes.js
  var require_lazysizes = __commonJS({
    "node_modules/lazysizes/lazysizes.js"(exports, module) {
      (function(window2, factory) {
        var lazySizes2 = factory(window2, window2.document, Date);
        window2.lazySizes = lazySizes2;
        if (typeof module == "object" && module.exports) {
          module.exports = lazySizes2;
        }
      })(
        typeof window != "undefined" ? window : {},
        /**
         * import("./types/global")
         * @typedef { import("./types/lazysizes-config").LazySizesConfigPartial } LazySizesConfigPartial
         */
        function l(window2, document2, Date2) {
          "use strict";
          var lazysizes, lazySizesCfg;
          (function() {
            var prop;
            var lazySizesDefaults = {
              lazyClass: "lazyload",
              loadedClass: "lazyloaded",
              loadingClass: "lazyloading",
              preloadClass: "lazypreload",
              errorClass: "lazyerror",
              //strictClass: 'lazystrict',
              autosizesClass: "lazyautosizes",
              fastLoadedClass: "ls-is-cached",
              iframeLoadMode: 0,
              srcAttr: "data-src",
              srcsetAttr: "data-srcset",
              sizesAttr: "data-sizes",
              //preloadAfterLoad: false,
              minSize: 40,
              customMedia: {},
              init: true,
              expFactor: 1.5,
              hFac: 0.8,
              loadMode: 2,
              loadHidden: true,
              ricTimeout: 0,
              throttleDelay: 125
            };
            lazySizesCfg = window2.lazySizesConfig || window2.lazysizesConfig || {};
            for (prop in lazySizesDefaults) {
              if (!(prop in lazySizesCfg)) {
                lazySizesCfg[prop] = lazySizesDefaults[prop];
              }
            }
          })();
          if (!document2 || !document2.getElementsByClassName) {
            return {
              init: function() {
              },
              /**
               * @type { LazySizesConfigPartial }
               */
              cfg: lazySizesCfg,
              /**
               * @type { true }
               */
              noSupport: true
            };
          }
          var docElem = document2.documentElement;
          var supportPicture = window2.HTMLPictureElement;
          var _addEventListener = "addEventListener";
          var _getAttribute = "getAttribute";
          var addEventListener = window2[_addEventListener].bind(window2);
          var setTimeout2 = window2.setTimeout;
          var requestAnimationFrame = window2.requestAnimationFrame || setTimeout2;
          var requestIdleCallback = window2.requestIdleCallback;
          var regPicture = /^picture$/i;
          var loadEvents = ["load", "error", "lazyincluded", "_lazyloaded"];
          var regClassCache = {};
          var forEach = Array.prototype.forEach;
          var hasClass = function(ele, cls) {
            if (!regClassCache[cls]) {
              regClassCache[cls] = new RegExp("(\\s|^)" + cls + "(\\s|$)");
            }
            return regClassCache[cls].test(ele[_getAttribute]("class") || "") && regClassCache[cls];
          };
          var addClass = function(ele, cls) {
            if (!hasClass(ele, cls)) {
              ele.setAttribute("class", (ele[_getAttribute]("class") || "").trim() + " " + cls);
            }
          };
          var removeClass = function(ele, cls) {
            var reg;
            if (reg = hasClass(ele, cls)) {
              ele.setAttribute("class", (ele[_getAttribute]("class") || "").replace(reg, " "));
            }
          };
          var addRemoveLoadEvents = function(dom, fn, add) {
            var action = add ? _addEventListener : "removeEventListener";
            if (add) {
              addRemoveLoadEvents(dom, fn);
            }
            loadEvents.forEach(function(evt) {
              dom[action](evt, fn);
            });
          };
          var triggerEvent = function(elem, name, detail, noBubbles, noCancelable) {
            var event = document2.createEvent("Event");
            if (!detail) {
              detail = {};
            }
            detail.instance = lazysizes;
            event.initEvent(name, !noBubbles, !noCancelable);
            event.detail = detail;
            elem.dispatchEvent(event);
            return event;
          };
          var updatePolyfill = function(el, full) {
            var polyfill;
            if (!supportPicture && (polyfill = window2.picturefill || lazySizesCfg.pf)) {
              if (full && full.src && !el[_getAttribute]("srcset")) {
                el.setAttribute("srcset", full.src);
              }
              polyfill({ reevaluate: true, elements: [el] });
            } else if (full && full.src) {
              el.src = full.src;
            }
          };
          var getCSS = function(elem, style) {
            return (getComputedStyle(elem, null) || {})[style];
          };
          var getWidth = function(elem, parent, width) {
            width = width || elem.offsetWidth;
            while (width < lazySizesCfg.minSize && parent && !elem._lazysizesWidth) {
              width = parent.offsetWidth;
              parent = parent.parentNode;
            }
            return width;
          };
          var rAF = function() {
            var running, waiting;
            var firstFns = [];
            var secondFns = [];
            var fns = firstFns;
            var run = function() {
              var runFns = fns;
              fns = firstFns.length ? secondFns : firstFns;
              running = true;
              waiting = false;
              while (runFns.length) {
                runFns.shift()();
              }
              running = false;
            };
            var rafBatch = function(fn, queue) {
              if (running && !queue) {
                fn.apply(this, arguments);
              } else {
                fns.push(fn);
                if (!waiting) {
                  waiting = true;
                  (document2.hidden ? setTimeout2 : requestAnimationFrame)(run);
                }
              }
            };
            rafBatch._lsFlush = run;
            return rafBatch;
          }();
          var rAFIt = function(fn, simple) {
            return simple ? function() {
              rAF(fn);
            } : function() {
              var that = this;
              var args = arguments;
              rAF(function() {
                fn.apply(that, args);
              });
            };
          };
          var throttle = function(fn) {
            var running;
            var lastTime = 0;
            var gDelay = lazySizesCfg.throttleDelay;
            var rICTimeout = lazySizesCfg.ricTimeout;
            var run = function() {
              running = false;
              lastTime = Date2.now();
              fn();
            };
            var idleCallback = requestIdleCallback && rICTimeout > 49 ? function() {
              requestIdleCallback(run, { timeout: rICTimeout });
              if (rICTimeout !== lazySizesCfg.ricTimeout) {
                rICTimeout = lazySizesCfg.ricTimeout;
              }
            } : rAFIt(function() {
              setTimeout2(run);
            }, true);
            return function(isPriority) {
              var delay;
              if (isPriority = isPriority === true) {
                rICTimeout = 33;
              }
              if (running) {
                return;
              }
              running = true;
              delay = gDelay - (Date2.now() - lastTime);
              if (delay < 0) {
                delay = 0;
              }
              if (isPriority || delay < 9) {
                idleCallback();
              } else {
                setTimeout2(idleCallback, delay);
              }
            };
          };
          var debounce = function(func) {
            var timeout, timestamp;
            var wait = 99;
            var run = function() {
              timeout = null;
              func();
            };
            var later = function() {
              var last = Date2.now() - timestamp;
              if (last < wait) {
                setTimeout2(later, wait - last);
              } else {
                (requestIdleCallback || run)(run);
              }
            };
            return function() {
              timestamp = Date2.now();
              if (!timeout) {
                timeout = setTimeout2(later, wait);
              }
            };
          };
          var loader = function() {
            var preloadElems, isCompleted, resetPreloadingTimer, loadMode, started;
            var eLvW, elvH, eLtop, eLleft, eLright, eLbottom, isBodyHidden;
            var regImg = /^img$/i;
            var regIframe = /^iframe$/i;
            var supportScroll = "onscroll" in window2 && !/(gle|ing)bot/.test(navigator.userAgent);
            var shrinkExpand = 0;
            var currentExpand = 0;
            var isLoading = 0;
            var lowRuns = -1;
            var resetPreloading = function(e2) {
              isLoading--;
              if (!e2 || isLoading < 0 || !e2.target) {
                isLoading = 0;
              }
            };
            var isVisible = function(elem) {
              if (isBodyHidden == null) {
                isBodyHidden = getCSS(document2.body, "visibility") == "hidden";
              }
              return isBodyHidden || !(getCSS(elem.parentNode, "visibility") == "hidden" && getCSS(elem, "visibility") == "hidden");
            };
            var isNestedVisible = function(elem, elemExpand) {
              var outerRect;
              var parent = elem;
              var visible = isVisible(elem);
              eLtop -= elemExpand;
              eLbottom += elemExpand;
              eLleft -= elemExpand;
              eLright += elemExpand;
              while (visible && (parent = parent.offsetParent) && parent != document2.body && parent != docElem) {
                visible = (getCSS(parent, "opacity") || 1) > 0;
                if (visible && getCSS(parent, "overflow") != "visible") {
                  outerRect = parent.getBoundingClientRect();
                  visible = eLright > outerRect.left && eLleft < outerRect.right && eLbottom > outerRect.top - 1 && eLtop < outerRect.bottom + 1;
                }
              }
              return visible;
            };
            var checkElements = function() {
              var eLlen, i3, rect, autoLoadElem, loadedSomething, elemExpand, elemNegativeExpand, elemExpandVal, beforeExpandVal, defaultExpand, preloadExpand, hFac;
              var lazyloadElems = lazysizes.elements;
              if ((loadMode = lazySizesCfg.loadMode) && isLoading < 8 && (eLlen = lazyloadElems.length)) {
                i3 = 0;
                lowRuns++;
                for (; i3 < eLlen; i3++) {
                  if (!lazyloadElems[i3] || lazyloadElems[i3]._lazyRace) {
                    continue;
                  }
                  if (!supportScroll || lazysizes.prematureUnveil && lazysizes.prematureUnveil(lazyloadElems[i3])) {
                    unveilElement(lazyloadElems[i3]);
                    continue;
                  }
                  if (!(elemExpandVal = lazyloadElems[i3][_getAttribute]("data-expand")) || !(elemExpand = elemExpandVal * 1)) {
                    elemExpand = currentExpand;
                  }
                  if (!defaultExpand) {
                    defaultExpand = !lazySizesCfg.expand || lazySizesCfg.expand < 1 ? docElem.clientHeight > 500 && docElem.clientWidth > 500 ? 500 : 370 : lazySizesCfg.expand;
                    lazysizes._defEx = defaultExpand;
                    preloadExpand = defaultExpand * lazySizesCfg.expFactor;
                    hFac = lazySizesCfg.hFac;
                    isBodyHidden = null;
                    if (currentExpand < preloadExpand && isLoading < 1 && lowRuns > 2 && loadMode > 2 && !document2.hidden) {
                      currentExpand = preloadExpand;
                      lowRuns = 0;
                    } else if (loadMode > 1 && lowRuns > 1 && isLoading < 6) {
                      currentExpand = defaultExpand;
                    } else {
                      currentExpand = shrinkExpand;
                    }
                  }
                  if (beforeExpandVal !== elemExpand) {
                    eLvW = innerWidth + elemExpand * hFac;
                    elvH = innerHeight + elemExpand;
                    elemNegativeExpand = elemExpand * -1;
                    beforeExpandVal = elemExpand;
                  }
                  rect = lazyloadElems[i3].getBoundingClientRect();
                  if ((eLbottom = rect.bottom) >= elemNegativeExpand && (eLtop = rect.top) <= elvH && (eLright = rect.right) >= elemNegativeExpand * hFac && (eLleft = rect.left) <= eLvW && (eLbottom || eLright || eLleft || eLtop) && (lazySizesCfg.loadHidden || isVisible(lazyloadElems[i3])) && (isCompleted && isLoading < 3 && !elemExpandVal && (loadMode < 3 || lowRuns < 4) || isNestedVisible(lazyloadElems[i3], elemExpand))) {
                    unveilElement(lazyloadElems[i3]);
                    loadedSomething = true;
                    if (isLoading > 9) {
                      break;
                    }
                  } else if (!loadedSomething && isCompleted && !autoLoadElem && isLoading < 4 && lowRuns < 4 && loadMode > 2 && (preloadElems[0] || lazySizesCfg.preloadAfterLoad) && (preloadElems[0] || !elemExpandVal && (eLbottom || eLright || eLleft || eLtop || lazyloadElems[i3][_getAttribute](lazySizesCfg.sizesAttr) != "auto"))) {
                    autoLoadElem = preloadElems[0] || lazyloadElems[i3];
                  }
                }
                if (autoLoadElem && !loadedSomething) {
                  unveilElement(autoLoadElem);
                }
              }
            };
            var throttledCheckElements = throttle(checkElements);
            var switchLoadingClass = function(e2) {
              var elem = e2.target;
              if (elem._lazyCache) {
                delete elem._lazyCache;
                return;
              }
              resetPreloading(e2);
              addClass(elem, lazySizesCfg.loadedClass);
              removeClass(elem, lazySizesCfg.loadingClass);
              addRemoveLoadEvents(elem, rafSwitchLoadingClass);
              triggerEvent(elem, "lazyloaded");
            };
            var rafedSwitchLoadingClass = rAFIt(switchLoadingClass);
            var rafSwitchLoadingClass = function(e2) {
              rafedSwitchLoadingClass({ target: e2.target });
            };
            var changeIframeSrc = function(elem, src) {
              var loadMode2 = elem.getAttribute("data-load-mode") || lazySizesCfg.iframeLoadMode;
              if (loadMode2 == 0) {
                elem.contentWindow.location.replace(src);
              } else if (loadMode2 == 1) {
                elem.src = src;
              }
            };
            var handleSources = function(source) {
              var customMedia;
              var sourceSrcset = source[_getAttribute](lazySizesCfg.srcsetAttr);
              if (customMedia = lazySizesCfg.customMedia[source[_getAttribute]("data-media") || source[_getAttribute]("media")]) {
                source.setAttribute("media", customMedia);
              }
              if (sourceSrcset) {
                source.setAttribute("srcset", sourceSrcset);
              }
            };
            var lazyUnveil = rAFIt(function(elem, detail, isAuto, sizes, isImg) {
              var src, srcset, parent, isPicture, event, firesLoad;
              if (!(event = triggerEvent(elem, "lazybeforeunveil", detail)).defaultPrevented) {
                if (sizes) {
                  if (isAuto) {
                    addClass(elem, lazySizesCfg.autosizesClass);
                  } else {
                    elem.setAttribute("sizes", sizes);
                  }
                }
                srcset = elem[_getAttribute](lazySizesCfg.srcsetAttr);
                src = elem[_getAttribute](lazySizesCfg.srcAttr);
                if (isImg) {
                  parent = elem.parentNode;
                  isPicture = parent && regPicture.test(parent.nodeName || "");
                }
                firesLoad = detail.firesLoad || "src" in elem && (srcset || src || isPicture);
                event = { target: elem };
                addClass(elem, lazySizesCfg.loadingClass);
                if (firesLoad) {
                  clearTimeout(resetPreloadingTimer);
                  resetPreloadingTimer = setTimeout2(resetPreloading, 2500);
                  addRemoveLoadEvents(elem, rafSwitchLoadingClass, true);
                }
                if (isPicture) {
                  forEach.call(parent.getElementsByTagName("source"), handleSources);
                }
                if (srcset) {
                  elem.setAttribute("srcset", srcset);
                } else if (src && !isPicture) {
                  if (regIframe.test(elem.nodeName)) {
                    changeIframeSrc(elem, src);
                  } else {
                    elem.src = src;
                  }
                }
                if (isImg && (srcset || isPicture)) {
                  updatePolyfill(elem, { src });
                }
              }
              if (elem._lazyRace) {
                delete elem._lazyRace;
              }
              removeClass(elem, lazySizesCfg.lazyClass);
              rAF(function() {
                var isLoaded = elem.complete && elem.naturalWidth > 1;
                if (!firesLoad || isLoaded) {
                  if (isLoaded) {
                    addClass(elem, lazySizesCfg.fastLoadedClass);
                  }
                  switchLoadingClass(event);
                  elem._lazyCache = true;
                  setTimeout2(function() {
                    if ("_lazyCache" in elem) {
                      delete elem._lazyCache;
                    }
                  }, 9);
                }
                if (elem.loading == "lazy") {
                  isLoading--;
                }
              }, true);
            });
            var unveilElement = function(elem) {
              if (elem._lazyRace) {
                return;
              }
              var detail;
              var isImg = regImg.test(elem.nodeName);
              var sizes = isImg && (elem[_getAttribute](lazySizesCfg.sizesAttr) || elem[_getAttribute]("sizes"));
              var isAuto = sizes == "auto";
              if ((isAuto || !isCompleted) && isImg && (elem[_getAttribute]("src") || elem.srcset) && !elem.complete && !hasClass(elem, lazySizesCfg.errorClass) && hasClass(elem, lazySizesCfg.lazyClass)) {
                return;
              }
              detail = triggerEvent(elem, "lazyunveilread").detail;
              if (isAuto) {
                autoSizer.updateElem(elem, true, elem.offsetWidth);
              }
              elem._lazyRace = true;
              isLoading++;
              lazyUnveil(elem, detail, isAuto, sizes, isImg);
            };
            var afterScroll = debounce(function() {
              lazySizesCfg.loadMode = 3;
              throttledCheckElements();
            });
            var altLoadmodeScrollListner = function() {
              if (lazySizesCfg.loadMode == 3) {
                lazySizesCfg.loadMode = 2;
              }
              afterScroll();
            };
            var onload = function() {
              if (isCompleted) {
                return;
              }
              if (Date2.now() - started < 999) {
                setTimeout2(onload, 999);
                return;
              }
              isCompleted = true;
              lazySizesCfg.loadMode = 3;
              throttledCheckElements();
              addEventListener("scroll", altLoadmodeScrollListner, true);
            };
            return {
              _: function() {
                started = Date2.now();
                lazysizes.elements = document2.getElementsByClassName(lazySizesCfg.lazyClass);
                preloadElems = document2.getElementsByClassName(lazySizesCfg.lazyClass + " " + lazySizesCfg.preloadClass);
                addEventListener("scroll", throttledCheckElements, true);
                addEventListener("resize", throttledCheckElements, true);
                addEventListener("pageshow", function(e2) {
                  if (e2.persisted) {
                    var loadingElements = document2.querySelectorAll("." + lazySizesCfg.loadingClass);
                    if (loadingElements.length && loadingElements.forEach) {
                      requestAnimationFrame(function() {
                        loadingElements.forEach(function(img) {
                          if (img.complete) {
                            unveilElement(img);
                          }
                        });
                      });
                    }
                  }
                });
                if (window2.MutationObserver) {
                  new MutationObserver(throttledCheckElements).observe(docElem, { childList: true, subtree: true, attributes: true });
                } else {
                  docElem[_addEventListener]("DOMNodeInserted", throttledCheckElements, true);
                  docElem[_addEventListener]("DOMAttrModified", throttledCheckElements, true);
                  setInterval(throttledCheckElements, 999);
                }
                addEventListener("hashchange", throttledCheckElements, true);
                ["focus", "mouseover", "click", "load", "transitionend", "animationend"].forEach(function(name) {
                  document2[_addEventListener](name, throttledCheckElements, true);
                });
                if (/d$|^c/.test(document2.readyState)) {
                  onload();
                } else {
                  addEventListener("load", onload);
                  document2[_addEventListener]("DOMContentLoaded", throttledCheckElements);
                  setTimeout2(onload, 2e4);
                }
                if (lazysizes.elements.length) {
                  checkElements();
                  rAF._lsFlush();
                } else {
                  throttledCheckElements();
                }
              },
              checkElems: throttledCheckElements,
              unveil: unveilElement,
              _aLSL: altLoadmodeScrollListner
            };
          }();
          var autoSizer = function() {
            var autosizesElems;
            var sizeElement = rAFIt(function(elem, parent, event, width) {
              var sources, i3, len;
              elem._lazysizesWidth = width;
              width += "px";
              elem.setAttribute("sizes", width);
              if (regPicture.test(parent.nodeName || "")) {
                sources = parent.getElementsByTagName("source");
                for (i3 = 0, len = sources.length; i3 < len; i3++) {
                  sources[i3].setAttribute("sizes", width);
                }
              }
              if (!event.detail.dataAttr) {
                updatePolyfill(elem, event.detail);
              }
            });
            var getSizeElement = function(elem, dataAttr, width) {
              var event;
              var parent = elem.parentNode;
              if (parent) {
                width = getWidth(elem, parent, width);
                event = triggerEvent(elem, "lazybeforesizes", { width, dataAttr: !!dataAttr });
                if (!event.defaultPrevented) {
                  width = event.detail.width;
                  if (width && width !== elem._lazysizesWidth) {
                    sizeElement(elem, parent, event, width);
                  }
                }
              }
            };
            var updateElementsSizes = function() {
              var i3;
              var len = autosizesElems.length;
              if (len) {
                i3 = 0;
                for (; i3 < len; i3++) {
                  getSizeElement(autosizesElems[i3]);
                }
              }
            };
            var debouncedUpdateElementsSizes = debounce(updateElementsSizes);
            return {
              _: function() {
                autosizesElems = document2.getElementsByClassName(lazySizesCfg.autosizesClass);
                addEventListener("resize", debouncedUpdateElementsSizes);
              },
              checkElems: debouncedUpdateElementsSizes,
              updateElem: getSizeElement
            };
          }();
          var init = function() {
            if (!init.i && document2.getElementsByClassName) {
              init.i = true;
              autoSizer._();
              loader._();
            }
          };
          setTimeout2(function() {
            if (lazySizesCfg.init) {
              init();
            }
          });
          lazysizes = {
            /**
             * @type { LazySizesConfigPartial }
             */
            cfg: lazySizesCfg,
            autoSizer,
            loader,
            init,
            uP: updatePolyfill,
            aC: addClass,
            rC: removeClass,
            hC: hasClass,
            fire: triggerEvent,
            gW: getWidth,
            rAF
          };
          return lazysizes;
        }
      );
    }
  });

  // node_modules/lazysizes/plugins/native-loading/ls.native-loading.js
  var require_ls_native_loading = __commonJS({
    "node_modules/lazysizes/plugins/native-loading/ls.native-loading.js"(exports, module) {
      (function(window2, factory) {
        var globalInstall = function() {
          factory(window2.lazySizes);
          window2.removeEventListener("lazyunveilread", globalInstall, true);
        };
        factory = factory.bind(null, window2, window2.document);
        if (typeof module == "object" && module.exports) {
          factory(require_lazysizes());
        } else if (typeof define == "function" && define.amd) {
          define(["lazysizes"], factory);
        } else if (window2.lazySizes) {
          globalInstall();
        } else {
          window2.addEventListener("lazyunveilread", globalInstall, true);
        }
      })(window, function(window2, document2, lazySizes2) {
        "use strict";
        var imgSupport = "loading" in HTMLImageElement.prototype;
        var iframeSupport = "loading" in HTMLIFrameElement.prototype;
        var isConfigSet = false;
        var oldPrematureUnveil = lazySizes2.prematureUnveil;
        var cfg = lazySizes2.cfg;
        var listenerMap = {
          focus: 1,
          mouseover: 1,
          click: 1,
          load: 1,
          transitionend: 1,
          animationend: 1,
          scroll: 1,
          resize: 1
        };
        if (!cfg.nativeLoading) {
          cfg.nativeLoading = {};
        }
        if (!window2.addEventListener || !window2.MutationObserver || !imgSupport && !iframeSupport) {
          return;
        }
        function disableEvents() {
          var loader = lazySizes2.loader;
          var throttledCheckElements = loader.checkElems;
          var removeALSL = function() {
            setTimeout(function() {
              window2.removeEventListener("scroll", loader._aLSL, true);
            }, 1e3);
          };
          var currentListenerMap = typeof cfg.nativeLoading.disableListeners == "object" ? cfg.nativeLoading.disableListeners : listenerMap;
          if (currentListenerMap.scroll) {
            window2.addEventListener("load", removeALSL);
            removeALSL();
            window2.removeEventListener("scroll", throttledCheckElements, true);
          }
          if (currentListenerMap.resize) {
            window2.removeEventListener("resize", throttledCheckElements, true);
          }
          Object.keys(currentListenerMap).forEach(function(name) {
            if (currentListenerMap[name]) {
              document2.removeEventListener(name, throttledCheckElements, true);
            }
          });
        }
        function runConfig() {
          if (isConfigSet) {
            return;
          }
          isConfigSet = true;
          if (imgSupport && iframeSupport && cfg.nativeLoading.disableListeners) {
            if (cfg.nativeLoading.disableListeners === true) {
              cfg.nativeLoading.setLoadingAttribute = true;
            }
            disableEvents();
          }
          if (cfg.nativeLoading.setLoadingAttribute) {
            window2.addEventListener("lazybeforeunveil", function(e2) {
              var element = e2.target;
              if ("loading" in element && !element.getAttribute("loading")) {
                element.setAttribute("loading", "lazy");
              }
            }, true);
          }
        }
        lazySizes2.prematureUnveil = function prematureUnveil(element) {
          if (!isConfigSet) {
            runConfig();
          }
          if ("loading" in element && (cfg.nativeLoading.setLoadingAttribute || element.getAttribute("loading")) && (element.getAttribute("data-sizes") != "auto" || element.offsetWidth)) {
            return true;
          }
          if (oldPrematureUnveil) {
            return oldPrematureUnveil(element);
          }
        };
      });
    }
  });

  // node_modules/clipboard/dist/clipboard.js
  var require_clipboard = __commonJS({
    "node_modules/clipboard/dist/clipboard.js"(exports, module) {
      (function webpackUniversalModuleDefinition(root, factory) {
        if (typeof exports === "object" && typeof module === "object")
          module.exports = factory();
        else if (typeof define === "function" && define.amd)
          define([], factory);
        else if (typeof exports === "object")
          exports["ClipboardJS"] = factory();
        else
          root["ClipboardJS"] = factory();
      })(exports, function() {
        return (
          /******/
          function() {
            var __webpack_modules__ = {
              /***/
              686: (
                /***/
                function(__unused_webpack_module, __webpack_exports__, __webpack_require__2) {
                  "use strict";
                  __webpack_require__2.d(__webpack_exports__, {
                    "default": function() {
                      return (
                        /* binding */
                        clipboard
                      );
                    }
                  });
                  var tiny_emitter = __webpack_require__2(279);
                  var tiny_emitter_default = /* @__PURE__ */ __webpack_require__2.n(tiny_emitter);
                  var listen = __webpack_require__2(370);
                  var listen_default = /* @__PURE__ */ __webpack_require__2.n(listen);
                  var src_select = __webpack_require__2(817);
                  var select_default = /* @__PURE__ */ __webpack_require__2.n(src_select);
                  ;
                  function command(type) {
                    try {
                      return document.execCommand(type);
                    } catch (err) {
                      return false;
                    }
                  }
                  ;
                  var ClipboardActionCut = function ClipboardActionCut2(target) {
                    var selectedText = select_default()(target);
                    command("cut");
                    return selectedText;
                  };
                  var actions_cut = ClipboardActionCut;
                  ;
                  function createFakeElement(value) {
                    var isRTL = document.documentElement.getAttribute("dir") === "rtl";
                    var fakeElement = document.createElement("textarea");
                    fakeElement.style.fontSize = "12pt";
                    fakeElement.style.border = "0";
                    fakeElement.style.padding = "0";
                    fakeElement.style.margin = "0";
                    fakeElement.style.position = "absolute";
                    fakeElement.style[isRTL ? "right" : "left"] = "-9999px";
                    var yPosition = window.pageYOffset || document.documentElement.scrollTop;
                    fakeElement.style.top = "".concat(yPosition, "px");
                    fakeElement.setAttribute("readonly", "");
                    fakeElement.value = value;
                    return fakeElement;
                  }
                  ;
                  var fakeCopyAction = function fakeCopyAction2(value, options) {
                    var fakeElement = createFakeElement(value);
                    options.container.appendChild(fakeElement);
                    var selectedText = select_default()(fakeElement);
                    command("copy");
                    fakeElement.remove();
                    return selectedText;
                  };
                  var ClipboardActionCopy = function ClipboardActionCopy2(target) {
                    var options = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : {
                      container: document.body
                    };
                    var selectedText = "";
                    if (typeof target === "string") {
                      selectedText = fakeCopyAction(target, options);
                    } else if (target instanceof HTMLInputElement && !["text", "search", "url", "tel", "password"].includes(target === null || target === void 0 ? void 0 : target.type)) {
                      selectedText = fakeCopyAction(target.value, options);
                    } else {
                      selectedText = select_default()(target);
                      command("copy");
                    }
                    return selectedText;
                  };
                  var actions_copy = ClipboardActionCopy;
                  ;
                  function _typeof(obj) {
                    "@babel/helpers - typeof";
                    if (typeof Symbol === "function" && typeof Symbol.iterator === "symbol") {
                      _typeof = function _typeof2(obj2) {
                        return typeof obj2;
                      };
                    } else {
                      _typeof = function _typeof2(obj2) {
                        return obj2 && typeof Symbol === "function" && obj2.constructor === Symbol && obj2 !== Symbol.prototype ? "symbol" : typeof obj2;
                      };
                    }
                    return _typeof(obj);
                  }
                  var ClipboardActionDefault = function ClipboardActionDefault2() {
                    var options = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : {};
                    var _options$action = options.action, action = _options$action === void 0 ? "copy" : _options$action, container = options.container, target = options.target, text = options.text;
                    if (action !== "copy" && action !== "cut") {
                      throw new Error('Invalid "action" value, use either "copy" or "cut"');
                    }
                    if (target !== void 0) {
                      if (target && _typeof(target) === "object" && target.nodeType === 1) {
                        if (action === "copy" && target.hasAttribute("disabled")) {
                          throw new Error('Invalid "target" attribute. Please use "readonly" instead of "disabled" attribute');
                        }
                        if (action === "cut" && (target.hasAttribute("readonly") || target.hasAttribute("disabled"))) {
                          throw new Error(`Invalid "target" attribute. You can't cut text from elements with "readonly" or "disabled" attributes`);
                        }
                      } else {
                        throw new Error('Invalid "target" value, use a valid Element');
                      }
                    }
                    if (text) {
                      return actions_copy(text, {
                        container
                      });
                    }
                    if (target) {
                      return action === "cut" ? actions_cut(target) : actions_copy(target, {
                        container
                      });
                    }
                  };
                  var actions_default = ClipboardActionDefault;
                  ;
                  function clipboard_typeof(obj) {
                    "@babel/helpers - typeof";
                    if (typeof Symbol === "function" && typeof Symbol.iterator === "symbol") {
                      clipboard_typeof = function _typeof2(obj2) {
                        return typeof obj2;
                      };
                    } else {
                      clipboard_typeof = function _typeof2(obj2) {
                        return obj2 && typeof Symbol === "function" && obj2.constructor === Symbol && obj2 !== Symbol.prototype ? "symbol" : typeof obj2;
                      };
                    }
                    return clipboard_typeof(obj);
                  }
                  function _classCallCheck(instance, Constructor) {
                    if (!(instance instanceof Constructor)) {
                      throw new TypeError("Cannot call a class as a function");
                    }
                  }
                  function _defineProperties(target, props) {
                    for (var i3 = 0; i3 < props.length; i3++) {
                      var descriptor = props[i3];
                      descriptor.enumerable = descriptor.enumerable || false;
                      descriptor.configurable = true;
                      if ("value" in descriptor) descriptor.writable = true;
                      Object.defineProperty(target, descriptor.key, descriptor);
                    }
                  }
                  function _createClass(Constructor, protoProps, staticProps) {
                    if (protoProps) _defineProperties(Constructor.prototype, protoProps);
                    if (staticProps) _defineProperties(Constructor, staticProps);
                    return Constructor;
                  }
                  function _inherits(subClass, superClass) {
                    if (typeof superClass !== "function" && superClass !== null) {
                      throw new TypeError("Super expression must either be null or a function");
                    }
                    subClass.prototype = Object.create(superClass && superClass.prototype, { constructor: { value: subClass, writable: true, configurable: true } });
                    if (superClass) _setPrototypeOf(subClass, superClass);
                  }
                  function _setPrototypeOf(o2, p) {
                    _setPrototypeOf = Object.setPrototypeOf || function _setPrototypeOf2(o3, p2) {
                      o3.__proto__ = p2;
                      return o3;
                    };
                    return _setPrototypeOf(o2, p);
                  }
                  function _createSuper(Derived) {
                    var hasNativeReflectConstruct = _isNativeReflectConstruct();
                    return function _createSuperInternal() {
                      var Super = _getPrototypeOf(Derived), result;
                      if (hasNativeReflectConstruct) {
                        var NewTarget = _getPrototypeOf(this).constructor;
                        result = Reflect.construct(Super, arguments, NewTarget);
                      } else {
                        result = Super.apply(this, arguments);
                      }
                      return _possibleConstructorReturn(this, result);
                    };
                  }
                  function _possibleConstructorReturn(self, call) {
                    if (call && (clipboard_typeof(call) === "object" || typeof call === "function")) {
                      return call;
                    }
                    return _assertThisInitialized(self);
                  }
                  function _assertThisInitialized(self) {
                    if (self === void 0) {
                      throw new ReferenceError("this hasn't been initialised - super() hasn't been called");
                    }
                    return self;
                  }
                  function _isNativeReflectConstruct() {
                    if (typeof Reflect === "undefined" || !Reflect.construct) return false;
                    if (Reflect.construct.sham) return false;
                    if (typeof Proxy === "function") return true;
                    try {
                      Date.prototype.toString.call(Reflect.construct(Date, [], function() {
                      }));
                      return true;
                    } catch (e2) {
                      return false;
                    }
                  }
                  function _getPrototypeOf(o2) {
                    _getPrototypeOf = Object.setPrototypeOf ? Object.getPrototypeOf : function _getPrototypeOf2(o3) {
                      return o3.__proto__ || Object.getPrototypeOf(o3);
                    };
                    return _getPrototypeOf(o2);
                  }
                  function getAttributeValue(suffix, element) {
                    var attribute = "data-clipboard-".concat(suffix);
                    if (!element.hasAttribute(attribute)) {
                      return;
                    }
                    return element.getAttribute(attribute);
                  }
                  var Clipboard2 = /* @__PURE__ */ function(_Emitter) {
                    _inherits(Clipboard3, _Emitter);
                    var _super = _createSuper(Clipboard3);
                    function Clipboard3(trigger, options) {
                      var _this;
                      _classCallCheck(this, Clipboard3);
                      _this = _super.call(this);
                      _this.resolveOptions(options);
                      _this.listenClick(trigger);
                      return _this;
                    }
                    _createClass(Clipboard3, [{
                      key: "resolveOptions",
                      value: function resolveOptions() {
                        var options = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : {};
                        this.action = typeof options.action === "function" ? options.action : this.defaultAction;
                        this.target = typeof options.target === "function" ? options.target : this.defaultTarget;
                        this.text = typeof options.text === "function" ? options.text : this.defaultText;
                        this.container = clipboard_typeof(options.container) === "object" ? options.container : document.body;
                      }
                      /**
                       * Adds a click event listener to the passed trigger.
                       * @param {String|HTMLElement|HTMLCollection|NodeList} trigger
                       */
                    }, {
                      key: "listenClick",
                      value: function listenClick(trigger) {
                        var _this2 = this;
                        this.listener = listen_default()(trigger, "click", function(e2) {
                          return _this2.onClick(e2);
                        });
                      }
                      /**
                       * Defines a new `ClipboardAction` on each click event.
                       * @param {Event} e
                       */
                    }, {
                      key: "onClick",
                      value: function onClick(e2) {
                        var trigger = e2.delegateTarget || e2.currentTarget;
                        var action = this.action(trigger) || "copy";
                        var text = actions_default({
                          action,
                          container: this.container,
                          target: this.target(trigger),
                          text: this.text(trigger)
                        });
                        this.emit(text ? "success" : "error", {
                          action,
                          text,
                          trigger,
                          clearSelection: function clearSelection() {
                            if (trigger) {
                              trigger.focus();
                            }
                            window.getSelection().removeAllRanges();
                          }
                        });
                      }
                      /**
                       * Default `action` lookup function.
                       * @param {Element} trigger
                       */
                    }, {
                      key: "defaultAction",
                      value: function defaultAction(trigger) {
                        return getAttributeValue("action", trigger);
                      }
                      /**
                       * Default `target` lookup function.
                       * @param {Element} trigger
                       */
                    }, {
                      key: "defaultTarget",
                      value: function defaultTarget(trigger) {
                        var selector = getAttributeValue("target", trigger);
                        if (selector) {
                          return document.querySelector(selector);
                        }
                      }
                      /**
                       * Allow fire programmatically a copy action
                       * @param {String|HTMLElement} target
                       * @param {Object} options
                       * @returns Text copied.
                       */
                    }, {
                      key: "defaultText",
                      /**
                       * Default `text` lookup function.
                       * @param {Element} trigger
                       */
                      value: function defaultText(trigger) {
                        return getAttributeValue("text", trigger);
                      }
                      /**
                       * Destroy lifecycle.
                       */
                    }, {
                      key: "destroy",
                      value: function destroy() {
                        this.listener.destroy();
                      }
                    }], [{
                      key: "copy",
                      value: function copy(target) {
                        var options = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : {
                          container: document.body
                        };
                        return actions_copy(target, options);
                      }
                      /**
                       * Allow fire programmatically a cut action
                       * @param {String|HTMLElement} target
                       * @returns Text cutted.
                       */
                    }, {
                      key: "cut",
                      value: function cut(target) {
                        return actions_cut(target);
                      }
                      /**
                       * Returns the support of the given action, or all actions if no action is
                       * given.
                       * @param {String} [action]
                       */
                    }, {
                      key: "isSupported",
                      value: function isSupported() {
                        var action = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : ["copy", "cut"];
                        var actions = typeof action === "string" ? [action] : action;
                        var support = !!document.queryCommandSupported;
                        actions.forEach(function(action2) {
                          support = support && !!document.queryCommandSupported(action2);
                        });
                        return support;
                      }
                    }]);
                    return Clipboard3;
                  }(tiny_emitter_default());
                  var clipboard = Clipboard2;
                }
              ),
              /***/
              828: (
                /***/
                function(module2) {
                  var DOCUMENT_NODE_TYPE = 9;
                  if (typeof Element !== "undefined" && !Element.prototype.matches) {
                    var proto = Element.prototype;
                    proto.matches = proto.matchesSelector || proto.mozMatchesSelector || proto.msMatchesSelector || proto.oMatchesSelector || proto.webkitMatchesSelector;
                  }
                  function closest(element, selector) {
                    while (element && element.nodeType !== DOCUMENT_NODE_TYPE) {
                      if (typeof element.matches === "function" && element.matches(selector)) {
                        return element;
                      }
                      element = element.parentNode;
                    }
                  }
                  module2.exports = closest;
                }
              ),
              /***/
              438: (
                /***/
                function(module2, __unused_webpack_exports, __webpack_require__2) {
                  var closest = __webpack_require__2(828);
                  function _delegate(element, selector, type, callback, useCapture) {
                    var listenerFn = listener.apply(this, arguments);
                    element.addEventListener(type, listenerFn, useCapture);
                    return {
                      destroy: function() {
                        element.removeEventListener(type, listenerFn, useCapture);
                      }
                    };
                  }
                  function delegate(elements, selector, type, callback, useCapture) {
                    if (typeof elements.addEventListener === "function") {
                      return _delegate.apply(null, arguments);
                    }
                    if (typeof type === "function") {
                      return _delegate.bind(null, document).apply(null, arguments);
                    }
                    if (typeof elements === "string") {
                      elements = document.querySelectorAll(elements);
                    }
                    return Array.prototype.map.call(elements, function(element) {
                      return _delegate(element, selector, type, callback, useCapture);
                    });
                  }
                  function listener(element, selector, type, callback) {
                    return function(e2) {
                      e2.delegateTarget = closest(e2.target, selector);
                      if (e2.delegateTarget) {
                        callback.call(element, e2);
                      }
                    };
                  }
                  module2.exports = delegate;
                }
              ),
              /***/
              879: (
                /***/
                function(__unused_webpack_module, exports2) {
                  exports2.node = function(value) {
                    return value !== void 0 && value instanceof HTMLElement && value.nodeType === 1;
                  };
                  exports2.nodeList = function(value) {
                    var type = Object.prototype.toString.call(value);
                    return value !== void 0 && (type === "[object NodeList]" || type === "[object HTMLCollection]") && "length" in value && (value.length === 0 || exports2.node(value[0]));
                  };
                  exports2.string = function(value) {
                    return typeof value === "string" || value instanceof String;
                  };
                  exports2.fn = function(value) {
                    var type = Object.prototype.toString.call(value);
                    return type === "[object Function]";
                  };
                }
              ),
              /***/
              370: (
                /***/
                function(module2, __unused_webpack_exports, __webpack_require__2) {
                  var is = __webpack_require__2(879);
                  var delegate = __webpack_require__2(438);
                  function listen(target, type, callback) {
                    if (!target && !type && !callback) {
                      throw new Error("Missing required arguments");
                    }
                    if (!is.string(type)) {
                      throw new TypeError("Second argument must be a String");
                    }
                    if (!is.fn(callback)) {
                      throw new TypeError("Third argument must be a Function");
                    }
                    if (is.node(target)) {
                      return listenNode(target, type, callback);
                    } else if (is.nodeList(target)) {
                      return listenNodeList(target, type, callback);
                    } else if (is.string(target)) {
                      return listenSelector(target, type, callback);
                    } else {
                      throw new TypeError("First argument must be a String, HTMLElement, HTMLCollection, or NodeList");
                    }
                  }
                  function listenNode(node, type, callback) {
                    node.addEventListener(type, callback);
                    return {
                      destroy: function() {
                        node.removeEventListener(type, callback);
                      }
                    };
                  }
                  function listenNodeList(nodeList, type, callback) {
                    Array.prototype.forEach.call(nodeList, function(node) {
                      node.addEventListener(type, callback);
                    });
                    return {
                      destroy: function() {
                        Array.prototype.forEach.call(nodeList, function(node) {
                          node.removeEventListener(type, callback);
                        });
                      }
                    };
                  }
                  function listenSelector(selector, type, callback) {
                    return delegate(document.body, selector, type, callback);
                  }
                  module2.exports = listen;
                }
              ),
              /***/
              817: (
                /***/
                function(module2) {
                  function select(element) {
                    var selectedText;
                    if (element.nodeName === "SELECT") {
                      element.focus();
                      selectedText = element.value;
                    } else if (element.nodeName === "INPUT" || element.nodeName === "TEXTAREA") {
                      var isReadOnly = element.hasAttribute("readonly");
                      if (!isReadOnly) {
                        element.setAttribute("readonly", "");
                      }
                      element.select();
                      element.setSelectionRange(0, element.value.length);
                      if (!isReadOnly) {
                        element.removeAttribute("readonly");
                      }
                      selectedText = element.value;
                    } else {
                      if (element.hasAttribute("contenteditable")) {
                        element.focus();
                      }
                      var selection = window.getSelection();
                      var range = document.createRange();
                      range.selectNodeContents(element);
                      selection.removeAllRanges();
                      selection.addRange(range);
                      selectedText = selection.toString();
                    }
                    return selectedText;
                  }
                  module2.exports = select;
                }
              ),
              /***/
              279: (
                /***/
                function(module2) {
                  function E() {
                  }
                  E.prototype = {
                    on: function(name, callback, ctx) {
                      var e2 = this.e || (this.e = {});
                      (e2[name] || (e2[name] = [])).push({
                        fn: callback,
                        ctx
                      });
                      return this;
                    },
                    once: function(name, callback, ctx) {
                      var self = this;
                      function listener() {
                        self.off(name, listener);
                        callback.apply(ctx, arguments);
                      }
                      ;
                      listener._ = callback;
                      return this.on(name, listener, ctx);
                    },
                    emit: function(name) {
                      var data = [].slice.call(arguments, 1);
                      var evtArr = ((this.e || (this.e = {}))[name] || []).slice();
                      var i3 = 0;
                      var len = evtArr.length;
                      for (i3; i3 < len; i3++) {
                        evtArr[i3].fn.apply(evtArr[i3].ctx, data);
                      }
                      return this;
                    },
                    off: function(name, callback) {
                      var e2 = this.e || (this.e = {});
                      var evts = e2[name];
                      var liveEvents = [];
                      if (evts && callback) {
                        for (var i3 = 0, len = evts.length; i3 < len; i3++) {
                          if (evts[i3].fn !== callback && evts[i3].fn._ !== callback)
                            liveEvents.push(evts[i3]);
                        }
                      }
                      liveEvents.length ? e2[name] = liveEvents : delete e2[name];
                      return this;
                    }
                  };
                  module2.exports = E;
                  module2.exports.TinyEmitter = E;
                }
              )
              /******/
            };
            var __webpack_module_cache__ = {};
            function __webpack_require__(moduleId) {
              if (__webpack_module_cache__[moduleId]) {
                return __webpack_module_cache__[moduleId].exports;
              }
              var module2 = __webpack_module_cache__[moduleId] = {
                /******/
                // no module.id needed
                /******/
                // no module.loaded needed
                /******/
                exports: {}
                /******/
              };
              __webpack_modules__[moduleId](module2, module2.exports, __webpack_require__);
              return module2.exports;
            }
            !function() {
              __webpack_require__.n = function(module2) {
                var getter = module2 && module2.__esModule ? (
                  /******/
                  function() {
                    return module2["default"];
                  }
                ) : (
                  /******/
                  function() {
                    return module2;
                  }
                );
                __webpack_require__.d(getter, { a: getter });
                return getter;
              };
            }();
            !function() {
              __webpack_require__.d = function(exports2, definition) {
                for (var key in definition) {
                  if (__webpack_require__.o(definition, key) && !__webpack_require__.o(exports2, key)) {
                    Object.defineProperty(exports2, key, { enumerable: true, get: definition[key] });
                  }
                }
              };
            }();
            !function() {
              __webpack_require__.o = function(obj, prop) {
                return Object.prototype.hasOwnProperty.call(obj, prop);
              };
            }();
            return __webpack_require__(686);
          }().default
        );
      });
    }
  });

  // ns-hugo:/Users/victor/Documents/Github/lematerial/lematerial-website/assets/js/custom.js
  var require_custom = __commonJS({
    "ns-hugo:/Users/victor/Documents/Github/lematerial/lematerial-website/assets/js/custom.js"(exports) {
      (() => __async(exports, null, function* () {
      }))();
    }
  });

  // node_modules/quicklink/dist/quicklink.mjs
  function e(e2) {
    return new Promise(function(n2, r2, t2) {
      (t2 = new XMLHttpRequest()).open("GET", e2, t2.withCredentials = true), t2.onload = function() {
        200 === t2.status ? n2() : r2();
      }, t2.send();
    });
  }
  var n;
  var r = (n = document.createElement("link")).relList && n.relList.supports && n.relList.supports("prefetch") ? function(e2) {
    return new Promise(function(n2, r2, t2) {
      (t2 = document.createElement("link")).rel = "prefetch", t2.href = e2, t2.onload = n2, t2.onerror = r2, document.head.appendChild(t2);
    });
  } : e;
  var t = window.requestIdleCallback || function(e2) {
    var n2 = Date.now();
    return setTimeout(function() {
      e2({ didTimeout: false, timeRemaining: function() {
        return Math.max(0, 50 - (Date.now() - n2));
      } });
    }, 1);
  };
  var o = /* @__PURE__ */ new Set();
  var i = /* @__PURE__ */ new Set();
  var c = false;
  function a(e2) {
    if (e2) {
      if (e2.saveData) return new Error("Save-Data is enabled");
      if (/2g/.test(e2.effectiveType)) return new Error("network conditions are poor");
    }
    return true;
  }
  function u(e2) {
    if (e2 || (e2 = {}), window.IntersectionObserver) {
      var n2 = function(e3) {
        e3 = e3 || 1;
        var n3 = [], r3 = 0;
        function t2() {
          r3 < e3 && n3.length > 0 && (n3.shift()(), r3++);
        }
        return [function(e4) {
          n3.push(e4) > 1 || t2();
        }, function() {
          r3--, t2();
        }];
      }(e2.throttle || 1 / 0), r2 = n2[0], a2 = n2[1], u2 = e2.limit || 1 / 0, l = e2.origins || [location.hostname], d = e2.ignores || [], h = e2.delay || 0, p = [], m = e2.timeoutFn || t, w = "function" == typeof e2.hrefFn && e2.hrefFn, g = e2.prerender || false;
      c = e2.prerenderAndPrefetch || false;
      var v = new IntersectionObserver(function(n3) {
        n3.forEach(function(n4) {
          if (n4.isIntersecting) p.push((n4 = n4.target).href), function(e3, n5) {
            n5 ? setTimeout(e3, n5) : e3();
          }(function() {
            -1 !== p.indexOf(n4.href) && (v.unobserve(n4), (c || g) && i.size < 1 ? f(w ? w(n4) : n4.href).catch(function(n5) {
              if (!e2.onError) throw n5;
              e2.onError(n5);
            }) : o.size < u2 && !g && r2(function() {
              s(w ? w(n4) : n4.href, e2.priority).then(a2).catch(function(n5) {
                a2(), e2.onError && e2.onError(n5);
              });
            }));
          }, h);
          else {
            var t2 = p.indexOf((n4 = n4.target).href);
            t2 > -1 && p.splice(t2);
          }
        });
      }, { threshold: e2.threshold || 0 });
      return m(function() {
        (e2.el || document).querySelectorAll("a").forEach(function(e3) {
          l.length && !l.includes(e3.hostname) || function e4(n3, r3) {
            return Array.isArray(r3) ? r3.some(function(r4) {
              return e4(n3, r4);
            }) : (r3.test || r3).call(r3, n3.href, n3);
          }(e3, d) || v.observe(e3);
        });
      }, { timeout: e2.timeout || 2e3 }), function() {
        o.clear(), v.disconnect();
      };
    }
  }
  function s(n2, t2, u2) {
    var s2 = a(navigator.connection);
    return s2 instanceof Error ? Promise.reject(new Error("Cannot prefetch, " + s2.message)) : (i.size > 0 && !c && console.warn("[Warning] You are using both prefetching and prerendering on the same document"), Promise.all([].concat(n2).map(function(n3) {
      if (!o.has(n3)) return o.add(n3), (t2 ? function(n4) {
        return window.fetch ? fetch(n4, { credentials: "include" }) : e(n4);
      } : r)(new URL(n3, location.href).toString());
    })));
  }
  function f(e2, n2) {
    var r2 = a(navigator.connection);
    if (r2 instanceof Error) return Promise.reject(new Error("Cannot prerender, " + r2.message));
    if (!HTMLScriptElement.supports("speculationrules")) return s(e2), Promise.reject(new Error("This browser does not support the speculation rules API. Falling back to prefetch."));
    if (document.querySelector('script[type="speculationrules"]')) return Promise.reject(new Error("Speculation Rules is already defined and cannot be altered."));
    for (var t2 = 0, u2 = [].concat(e2); t2 < u2.length; t2 += 1) {
      var f2 = u2[t2];
      if (window.location.origin !== new URL(f2, window.location.href).origin) return Promise.reject(new Error("Only same origin URLs are allowed: " + f2));
      i.add(f2);
    }
    o.size > 0 && !c && console.warn("[Warning] You are using both prefetching and prerendering on the same document");
    var l = function(e3) {
      var n3 = document.createElement("script");
      n3.type = "speculationrules", n3.text = '{"prerender":[{"source": "list","urls": ["' + Array.from(e3).join('","') + '"]}]}';
      try {
        document.head.appendChild(n3);
      } catch (e4) {
        return e4;
      }
      return true;
    }(i);
    return true === l ? Promise.resolve() : Promise.reject(l);
  }

  // node_modules/@thulite/core/assets/js/core.js
  var import_lazysizes = __toESM(require_lazysizes());
  var import_ls = __toESM(require_ls_native_loading());
  u({
    ignores: [
      /\/api\/?/,
      (uri) => uri.includes(".zip"),
      (uri, elem) => elem.hasAttribute("noprefetch"),
      (uri, elem) => elem.hash && elem.pathname === window.location.pathname
    ]
  });
  import_lazysizes.default.cfg.nativeLoading = {
    setLoadingAttribute: true,
    disableListeners: {
      scroll: true
    }
  };

  // ns-hugo:/Users/victor/Documents/Github/lematerial/lematerial-website/node_modules/@thulite/doks-core/assets/js/clipboard.js
  var import_clipboard = __toESM(require_clipboard());
  (() => {
    "use strict";
    var cb = document.getElementsByClassName("highlight");
    for (var i3 = 0; i3 < cb.length; ++i3) {
      var element = cb[i3];
      element.insertAdjacentHTML("afterbegin", '<div class="copy"><button title="Copy to clipboard" class="btn-copy" aria-label="Clipboard button"><div></div></button></div>');
    }
    var clipboard = new import_clipboard.default(".btn-copy", {
      target: function(trigger) {
        return trigger.parentNode.nextElementSibling;
      }
    });
    clipboard.on("success", function(e2) {
      e2.clearSelection();
    });
    clipboard.on("error", function(e2) {
      console.error("Action:", e2.action);
      console.error("Trigger:", e2.trigger);
    });
  })();

  // ns-hugo:/Users/victor/Documents/Github/lematerial/lematerial-website/node_modules/@thulite/doks-core/assets/js/to-top.js
  var topButton = document.getElementById("toTop");
  if (topButton !== null) {
    topButton.classList.remove("fade");
    window.onscroll = function() {
      scrollFunction();
    };
    topButton.addEventListener("click", topFunction);
  }
  function scrollFunction() {
    if (document.body.scrollTop > 270 || document.documentElement.scrollTop > 270) {
      topButton.classList.add("fade");
    } else {
      topButton.classList.remove("fade");
    }
  }
  function topFunction() {
    document.body.scrollTop = 0;
    document.documentElement.scrollTop = 0;
  }

  // ns-hugo:/Users/victor/Documents/Github/lematerial/lematerial-website/node_modules/@thulite/doks-core/assets/js/tabs.js
  var i2;
  var allTabs = document.querySelectorAll("[data-toggle-tab]");
  var allPanes = document.querySelectorAll("[data-pane]");
  function toggleTabs(event) {
    if (event.target) {
      event.preventDefault();
      var clickedTab = event.currentTarget;
      var targetKey = clickedTab.getAttribute("data-toggle-tab");
    } else {
      var targetKey = event;
    }
    if (window.localStorage) {
      window.localStorage.setItem("configLangPref", targetKey);
    }
    var selectedTabs = document.querySelectorAll("[data-toggle-tab=" + targetKey + "]");
    var selectedPanes = document.querySelectorAll("[data-pane=" + targetKey + "]");
    for (var i3 = 0; i3 < allTabs.length; i3++) {
      allTabs[i3].classList.remove("active");
      allPanes[i3].classList.remove("active");
    }
    for (var i3 = 0; i3 < selectedTabs.length; i3++) {
      selectedTabs[i3].classList.add("active");
      selectedPanes[i3].classList.add("show", "active");
    }
  }
  for (i2 = 0; i2 < allTabs.length; i2++) {
    allTabs[i2].addEventListener("click", toggleTabs);
  }
  if (window.localStorage.getItem("configLangPref")) {
    toggleTabs(window.localStorage.getItem("configLangPref"));
  }

  // <stdin>
  var import_custom = __toESM(require_custom());
})();
/*! Bundled license information:

clipboard/dist/clipboard.js:
  (*!
   * clipboard.js v2.0.11
   * https://clipboardjs.com/
   *
   * Licensed MIT © Zeno Rocha
   *)

@thulite/doks-core/assets/js/clipboard.js:
  (*!
   * clipboard.js for Bootstrap based Thulite sites
   * Copyright 2021-2024 Thulite
   * Licensed under the MIT License
   *)
*/
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsibm9kZV9tb2R1bGVzL2xhenlzaXplcy9sYXp5c2l6ZXMuanMiLCAibm9kZV9tb2R1bGVzL2xhenlzaXplcy9wbHVnaW5zL25hdGl2ZS1sb2FkaW5nL2xzLm5hdGl2ZS1sb2FkaW5nLmpzIiwgIm5vZGVfbW9kdWxlcy9jbGlwYm9hcmQvZGlzdC9jbGlwYm9hcmQuanMiLCAibnMtaHVnbzovVXNlcnMvdmljdG9yL0RvY3VtZW50cy9HaXRodWIvbGVtYXRlcmlhbC9sZW1hdGVyaWFsLXdlYnNpdGUvYXNzZXRzL2pzL2N1c3RvbS5qcyIsICJub2RlX21vZHVsZXMvcXVpY2tsaW5rL2Rpc3QvcXVpY2tsaW5rLm1qcyIsICJub2RlX21vZHVsZXMvQHRodWxpdGUvY29yZS9hc3NldHMvanMvY29yZS5qcyIsICJucy1odWdvOi9Vc2Vycy92aWN0b3IvRG9jdW1lbnRzL0dpdGh1Yi9sZW1hdGVyaWFsL2xlbWF0ZXJpYWwtd2Vic2l0ZS9ub2RlX21vZHVsZXMvQHRodWxpdGUvZG9rcy1jb3JlL2Fzc2V0cy9qcy9jbGlwYm9hcmQuanMiLCAibnMtaHVnbzovVXNlcnMvdmljdG9yL0RvY3VtZW50cy9HaXRodWIvbGVtYXRlcmlhbC9sZW1hdGVyaWFsLXdlYnNpdGUvbm9kZV9tb2R1bGVzL0B0aHVsaXRlL2Rva3MtY29yZS9hc3NldHMvanMvdG8tdG9wLmpzIiwgIm5zLWh1Z286L1VzZXJzL3ZpY3Rvci9Eb2N1bWVudHMvR2l0aHViL2xlbWF0ZXJpYWwvbGVtYXRlcmlhbC13ZWJzaXRlL25vZGVfbW9kdWxlcy9AdGh1bGl0ZS9kb2tzLWNvcmUvYXNzZXRzL2pzL3RhYnMuanMiLCAiPHN0ZGluPiJdLAogICJzb3VyY2VzQ29udGVudCI6IFsiKGZ1bmN0aW9uKHdpbmRvdywgZmFjdG9yeSkge1xuXHR2YXIgbGF6eVNpemVzID0gZmFjdG9yeSh3aW5kb3csIHdpbmRvdy5kb2N1bWVudCwgRGF0ZSk7XG5cdHdpbmRvdy5sYXp5U2l6ZXMgPSBsYXp5U2l6ZXM7XG5cdGlmKHR5cGVvZiBtb2R1bGUgPT0gJ29iamVjdCcgJiYgbW9kdWxlLmV4cG9ydHMpe1xuXHRcdG1vZHVsZS5leHBvcnRzID0gbGF6eVNpemVzO1xuXHR9XG59KHR5cGVvZiB3aW5kb3cgIT0gJ3VuZGVmaW5lZCcgP1xuICAgICAgd2luZG93IDoge30sIFxuLyoqXG4gKiBpbXBvcnQoXCIuL3R5cGVzL2dsb2JhbFwiKVxuICogQHR5cGVkZWYgeyBpbXBvcnQoXCIuL3R5cGVzL2xhenlzaXplcy1jb25maWdcIikuTGF6eVNpemVzQ29uZmlnUGFydGlhbCB9IExhenlTaXplc0NvbmZpZ1BhcnRpYWxcbiAqL1xuZnVuY3Rpb24gbCh3aW5kb3csIGRvY3VtZW50LCBEYXRlKSB7IC8vIFBhc3MgaW4gdGhlIHdpbmRvdyBEYXRlIGZ1bmN0aW9uIGFsc28gZm9yIFNTUiBiZWNhdXNlIHRoZSBEYXRlIGNsYXNzIGNhbiBiZSBsb3N0XG5cdCd1c2Ugc3RyaWN0Jztcblx0Lypqc2hpbnQgZXFudWxsOnRydWUgKi9cblxuXHR2YXIgbGF6eXNpemVzLFxuXHRcdC8qKlxuXHRcdCAqIEB0eXBlIHsgTGF6eVNpemVzQ29uZmlnUGFydGlhbCB9XG5cdFx0ICovXG5cdFx0bGF6eVNpemVzQ2ZnO1xuXG5cdChmdW5jdGlvbigpe1xuXHRcdHZhciBwcm9wO1xuXG5cdFx0dmFyIGxhenlTaXplc0RlZmF1bHRzID0ge1xuXHRcdFx0bGF6eUNsYXNzOiAnbGF6eWxvYWQnLFxuXHRcdFx0bG9hZGVkQ2xhc3M6ICdsYXp5bG9hZGVkJyxcblx0XHRcdGxvYWRpbmdDbGFzczogJ2xhenlsb2FkaW5nJyxcblx0XHRcdHByZWxvYWRDbGFzczogJ2xhenlwcmVsb2FkJyxcblx0XHRcdGVycm9yQ2xhc3M6ICdsYXp5ZXJyb3InLFxuXHRcdFx0Ly9zdHJpY3RDbGFzczogJ2xhenlzdHJpY3QnLFxuXHRcdFx0YXV0b3NpemVzQ2xhc3M6ICdsYXp5YXV0b3NpemVzJyxcblx0XHRcdGZhc3RMb2FkZWRDbGFzczogJ2xzLWlzLWNhY2hlZCcsXG5cdFx0XHRpZnJhbWVMb2FkTW9kZTogMCxcblx0XHRcdHNyY0F0dHI6ICdkYXRhLXNyYycsXG5cdFx0XHRzcmNzZXRBdHRyOiAnZGF0YS1zcmNzZXQnLFxuXHRcdFx0c2l6ZXNBdHRyOiAnZGF0YS1zaXplcycsXG5cdFx0XHQvL3ByZWxvYWRBZnRlckxvYWQ6IGZhbHNlLFxuXHRcdFx0bWluU2l6ZTogNDAsXG5cdFx0XHRjdXN0b21NZWRpYToge30sXG5cdFx0XHRpbml0OiB0cnVlLFxuXHRcdFx0ZXhwRmFjdG9yOiAxLjUsXG5cdFx0XHRoRmFjOiAwLjgsXG5cdFx0XHRsb2FkTW9kZTogMixcblx0XHRcdGxvYWRIaWRkZW46IHRydWUsXG5cdFx0XHRyaWNUaW1lb3V0OiAwLFxuXHRcdFx0dGhyb3R0bGVEZWxheTogMTI1LFxuXHRcdH07XG5cblx0XHRsYXp5U2l6ZXNDZmcgPSB3aW5kb3cubGF6eVNpemVzQ29uZmlnIHx8IHdpbmRvdy5sYXp5c2l6ZXNDb25maWcgfHwge307XG5cblx0XHRmb3IocHJvcCBpbiBsYXp5U2l6ZXNEZWZhdWx0cyl7XG5cdFx0XHRpZighKHByb3AgaW4gbGF6eVNpemVzQ2ZnKSl7XG5cdFx0XHRcdGxhenlTaXplc0NmZ1twcm9wXSA9IGxhenlTaXplc0RlZmF1bHRzW3Byb3BdO1xuXHRcdFx0fVxuXHRcdH1cblx0fSkoKTtcblxuXHRpZiAoIWRvY3VtZW50IHx8ICFkb2N1bWVudC5nZXRFbGVtZW50c0J5Q2xhc3NOYW1lKSB7XG5cdFx0cmV0dXJuIHtcblx0XHRcdGluaXQ6IGZ1bmN0aW9uICgpIHt9LFxuXHRcdFx0LyoqXG5cdFx0XHQgKiBAdHlwZSB7IExhenlTaXplc0NvbmZpZ1BhcnRpYWwgfVxuXHRcdFx0ICovXG5cdFx0XHRjZmc6IGxhenlTaXplc0NmZyxcblx0XHRcdC8qKlxuXHRcdFx0ICogQHR5cGUgeyB0cnVlIH1cblx0XHRcdCAqL1xuXHRcdFx0bm9TdXBwb3J0OiB0cnVlLFxuXHRcdH07XG5cdH1cblxuXHR2YXIgZG9jRWxlbSA9IGRvY3VtZW50LmRvY3VtZW50RWxlbWVudDtcblxuXHR2YXIgc3VwcG9ydFBpY3R1cmUgPSB3aW5kb3cuSFRNTFBpY3R1cmVFbGVtZW50O1xuXG5cdHZhciBfYWRkRXZlbnRMaXN0ZW5lciA9ICdhZGRFdmVudExpc3RlbmVyJztcblxuXHR2YXIgX2dldEF0dHJpYnV0ZSA9ICdnZXRBdHRyaWJ1dGUnO1xuXG5cdC8qKlxuXHQgKiBVcGRhdGUgdG8gYmluZCB0byB3aW5kb3cgYmVjYXVzZSAndGhpcycgYmVjb21lcyBudWxsIGR1cmluZyBTU1Jcblx0ICogYnVpbGRzLlxuXHQgKi9cblx0dmFyIGFkZEV2ZW50TGlzdGVuZXIgPSB3aW5kb3dbX2FkZEV2ZW50TGlzdGVuZXJdLmJpbmQod2luZG93KTtcblxuXHR2YXIgc2V0VGltZW91dCA9IHdpbmRvdy5zZXRUaW1lb3V0O1xuXG5cdHZhciByZXF1ZXN0QW5pbWF0aW9uRnJhbWUgPSB3aW5kb3cucmVxdWVzdEFuaW1hdGlvbkZyYW1lIHx8IHNldFRpbWVvdXQ7XG5cblx0dmFyIHJlcXVlc3RJZGxlQ2FsbGJhY2sgPSB3aW5kb3cucmVxdWVzdElkbGVDYWxsYmFjaztcblxuXHR2YXIgcmVnUGljdHVyZSA9IC9ecGljdHVyZSQvaTtcblxuXHR2YXIgbG9hZEV2ZW50cyA9IFsnbG9hZCcsICdlcnJvcicsICdsYXp5aW5jbHVkZWQnLCAnX2xhenlsb2FkZWQnXTtcblxuXHR2YXIgcmVnQ2xhc3NDYWNoZSA9IHt9O1xuXG5cdHZhciBmb3JFYWNoID0gQXJyYXkucHJvdG90eXBlLmZvckVhY2g7XG5cblx0LyoqXG5cdCAqIEBwYXJhbSBlbGUge0VsZW1lbnR9XG5cdCAqIEBwYXJhbSBjbHMge3N0cmluZ31cblx0ICovXG5cdHZhciBoYXNDbGFzcyA9IGZ1bmN0aW9uKGVsZSwgY2xzKSB7XG5cdFx0aWYoIXJlZ0NsYXNzQ2FjaGVbY2xzXSl7XG5cdFx0XHRyZWdDbGFzc0NhY2hlW2Nsc10gPSBuZXcgUmVnRXhwKCcoXFxcXHN8XiknK2NscysnKFxcXFxzfCQpJyk7XG5cdFx0fVxuXHRcdHJldHVybiByZWdDbGFzc0NhY2hlW2Nsc10udGVzdChlbGVbX2dldEF0dHJpYnV0ZV0oJ2NsYXNzJykgfHwgJycpICYmIHJlZ0NsYXNzQ2FjaGVbY2xzXTtcblx0fTtcblxuXHQvKipcblx0ICogQHBhcmFtIGVsZSB7RWxlbWVudH1cblx0ICogQHBhcmFtIGNscyB7c3RyaW5nfVxuXHQgKi9cblx0dmFyIGFkZENsYXNzID0gZnVuY3Rpb24oZWxlLCBjbHMpIHtcblx0XHRpZiAoIWhhc0NsYXNzKGVsZSwgY2xzKSl7XG5cdFx0XHRlbGUuc2V0QXR0cmlidXRlKCdjbGFzcycsIChlbGVbX2dldEF0dHJpYnV0ZV0oJ2NsYXNzJykgfHwgJycpLnRyaW0oKSArICcgJyArIGNscyk7XG5cdFx0fVxuXHR9O1xuXG5cdC8qKlxuXHQgKiBAcGFyYW0gZWxlIHtFbGVtZW50fVxuXHQgKiBAcGFyYW0gY2xzIHtzdHJpbmd9XG5cdCAqL1xuXHR2YXIgcmVtb3ZlQ2xhc3MgPSBmdW5jdGlvbihlbGUsIGNscykge1xuXHRcdHZhciByZWc7XG5cdFx0aWYgKChyZWcgPSBoYXNDbGFzcyhlbGUsY2xzKSkpIHtcblx0XHRcdGVsZS5zZXRBdHRyaWJ1dGUoJ2NsYXNzJywgKGVsZVtfZ2V0QXR0cmlidXRlXSgnY2xhc3MnKSB8fCAnJykucmVwbGFjZShyZWcsICcgJykpO1xuXHRcdH1cblx0fTtcblxuXHR2YXIgYWRkUmVtb3ZlTG9hZEV2ZW50cyA9IGZ1bmN0aW9uKGRvbSwgZm4sIGFkZCl7XG5cdFx0dmFyIGFjdGlvbiA9IGFkZCA/IF9hZGRFdmVudExpc3RlbmVyIDogJ3JlbW92ZUV2ZW50TGlzdGVuZXInO1xuXHRcdGlmKGFkZCl7XG5cdFx0XHRhZGRSZW1vdmVMb2FkRXZlbnRzKGRvbSwgZm4pO1xuXHRcdH1cblx0XHRsb2FkRXZlbnRzLmZvckVhY2goZnVuY3Rpb24oZXZ0KXtcblx0XHRcdGRvbVthY3Rpb25dKGV2dCwgZm4pO1xuXHRcdH0pO1xuXHR9O1xuXG5cdC8qKlxuXHQgKiBAcGFyYW0gZWxlbSB7IEVsZW1lbnQgfVxuXHQgKiBAcGFyYW0gbmFtZSB7IHN0cmluZyB9XG5cdCAqIEBwYXJhbSBkZXRhaWwgeyBhbnkgfVxuXHQgKiBAcGFyYW0gbm9CdWJibGVzIHsgYm9vbGVhbiB9XG5cdCAqIEBwYXJhbSBub0NhbmNlbGFibGUgeyBib29sZWFuIH1cblx0ICogQHJldHVybnMgeyBDdXN0b21FdmVudCB9XG5cdCAqL1xuXHR2YXIgdHJpZ2dlckV2ZW50ID0gZnVuY3Rpb24oZWxlbSwgbmFtZSwgZGV0YWlsLCBub0J1YmJsZXMsIG5vQ2FuY2VsYWJsZSl7XG5cdFx0dmFyIGV2ZW50ID0gZG9jdW1lbnQuY3JlYXRlRXZlbnQoJ0V2ZW50Jyk7XG5cblx0XHRpZighZGV0YWlsKXtcblx0XHRcdGRldGFpbCA9IHt9O1xuXHRcdH1cblxuXHRcdGRldGFpbC5pbnN0YW5jZSA9IGxhenlzaXplcztcblxuXHRcdGV2ZW50LmluaXRFdmVudChuYW1lLCAhbm9CdWJibGVzLCAhbm9DYW5jZWxhYmxlKTtcblxuXHRcdGV2ZW50LmRldGFpbCA9IGRldGFpbDtcblxuXHRcdGVsZW0uZGlzcGF0Y2hFdmVudChldmVudCk7XG5cdFx0cmV0dXJuIGV2ZW50O1xuXHR9O1xuXG5cdHZhciB1cGRhdGVQb2x5ZmlsbCA9IGZ1bmN0aW9uIChlbCwgZnVsbCl7XG5cdFx0dmFyIHBvbHlmaWxsO1xuXHRcdGlmKCAhc3VwcG9ydFBpY3R1cmUgJiYgKCBwb2x5ZmlsbCA9ICh3aW5kb3cucGljdHVyZWZpbGwgfHwgbGF6eVNpemVzQ2ZnLnBmKSApICl7XG5cdFx0XHRpZihmdWxsICYmIGZ1bGwuc3JjICYmICFlbFtfZ2V0QXR0cmlidXRlXSgnc3Jjc2V0Jykpe1xuXHRcdFx0XHRlbC5zZXRBdHRyaWJ1dGUoJ3NyY3NldCcsIGZ1bGwuc3JjKTtcblx0XHRcdH1cblx0XHRcdHBvbHlmaWxsKHtyZWV2YWx1YXRlOiB0cnVlLCBlbGVtZW50czogW2VsXX0pO1xuXHRcdH0gZWxzZSBpZihmdWxsICYmIGZ1bGwuc3JjKXtcblx0XHRcdGVsLnNyYyA9IGZ1bGwuc3JjO1xuXHRcdH1cblx0fTtcblxuXHR2YXIgZ2V0Q1NTID0gZnVuY3Rpb24gKGVsZW0sIHN0eWxlKXtcblx0XHRyZXR1cm4gKGdldENvbXB1dGVkU3R5bGUoZWxlbSwgbnVsbCkgfHwge30pW3N0eWxlXTtcblx0fTtcblxuXHQvKipcblx0ICpcblx0ICogQHBhcmFtIGVsZW0geyBFbGVtZW50IH1cblx0ICogQHBhcmFtIHBhcmVudCB7IEVsZW1lbnQgfVxuXHQgKiBAcGFyYW0gW3dpZHRoXSB7bnVtYmVyfVxuXHQgKiBAcmV0dXJucyB7bnVtYmVyfVxuXHQgKi9cblx0dmFyIGdldFdpZHRoID0gZnVuY3Rpb24oZWxlbSwgcGFyZW50LCB3aWR0aCl7XG5cdFx0d2lkdGggPSB3aWR0aCB8fCBlbGVtLm9mZnNldFdpZHRoO1xuXG5cdFx0d2hpbGUod2lkdGggPCBsYXp5U2l6ZXNDZmcubWluU2l6ZSAmJiBwYXJlbnQgJiYgIWVsZW0uX2xhenlzaXplc1dpZHRoKXtcblx0XHRcdHdpZHRoID0gIHBhcmVudC5vZmZzZXRXaWR0aDtcblx0XHRcdHBhcmVudCA9IHBhcmVudC5wYXJlbnROb2RlO1xuXHRcdH1cblxuXHRcdHJldHVybiB3aWR0aDtcblx0fTtcblxuXHR2YXIgckFGID0gKGZ1bmN0aW9uKCl7XG5cdFx0dmFyIHJ1bm5pbmcsIHdhaXRpbmc7XG5cdFx0dmFyIGZpcnN0Rm5zID0gW107XG5cdFx0dmFyIHNlY29uZEZucyA9IFtdO1xuXHRcdHZhciBmbnMgPSBmaXJzdEZucztcblxuXHRcdHZhciBydW4gPSBmdW5jdGlvbigpe1xuXHRcdFx0dmFyIHJ1bkZucyA9IGZucztcblxuXHRcdFx0Zm5zID0gZmlyc3RGbnMubGVuZ3RoID8gc2Vjb25kRm5zIDogZmlyc3RGbnM7XG5cblx0XHRcdHJ1bm5pbmcgPSB0cnVlO1xuXHRcdFx0d2FpdGluZyA9IGZhbHNlO1xuXG5cdFx0XHR3aGlsZShydW5GbnMubGVuZ3RoKXtcblx0XHRcdFx0cnVuRm5zLnNoaWZ0KCkoKTtcblx0XHRcdH1cblxuXHRcdFx0cnVubmluZyA9IGZhbHNlO1xuXHRcdH07XG5cblx0XHR2YXIgcmFmQmF0Y2ggPSBmdW5jdGlvbihmbiwgcXVldWUpe1xuXHRcdFx0aWYocnVubmluZyAmJiAhcXVldWUpe1xuXHRcdFx0XHRmbi5hcHBseSh0aGlzLCBhcmd1bWVudHMpO1xuXHRcdFx0fSBlbHNlIHtcblx0XHRcdFx0Zm5zLnB1c2goZm4pO1xuXG5cdFx0XHRcdGlmKCF3YWl0aW5nKXtcblx0XHRcdFx0XHR3YWl0aW5nID0gdHJ1ZTtcblx0XHRcdFx0XHQoZG9jdW1lbnQuaGlkZGVuID8gc2V0VGltZW91dCA6IHJlcXVlc3RBbmltYXRpb25GcmFtZSkocnVuKTtcblx0XHRcdFx0fVxuXHRcdFx0fVxuXHRcdH07XG5cblx0XHRyYWZCYXRjaC5fbHNGbHVzaCA9IHJ1bjtcblxuXHRcdHJldHVybiByYWZCYXRjaDtcblx0fSkoKTtcblxuXHR2YXIgckFGSXQgPSBmdW5jdGlvbihmbiwgc2ltcGxlKXtcblx0XHRyZXR1cm4gc2ltcGxlID9cblx0XHRcdGZ1bmN0aW9uKCkge1xuXHRcdFx0XHRyQUYoZm4pO1xuXHRcdFx0fSA6XG5cdFx0XHRmdW5jdGlvbigpe1xuXHRcdFx0XHR2YXIgdGhhdCA9IHRoaXM7XG5cdFx0XHRcdHZhciBhcmdzID0gYXJndW1lbnRzO1xuXHRcdFx0XHRyQUYoZnVuY3Rpb24oKXtcblx0XHRcdFx0XHRmbi5hcHBseSh0aGF0LCBhcmdzKTtcblx0XHRcdFx0fSk7XG5cdFx0XHR9XG5cdFx0O1xuXHR9O1xuXG5cdHZhciB0aHJvdHRsZSA9IGZ1bmN0aW9uKGZuKXtcblx0XHR2YXIgcnVubmluZztcblx0XHR2YXIgbGFzdFRpbWUgPSAwO1xuXHRcdHZhciBnRGVsYXkgPSBsYXp5U2l6ZXNDZmcudGhyb3R0bGVEZWxheTtcblx0XHR2YXIgcklDVGltZW91dCA9IGxhenlTaXplc0NmZy5yaWNUaW1lb3V0O1xuXHRcdHZhciBydW4gPSBmdW5jdGlvbigpe1xuXHRcdFx0cnVubmluZyA9IGZhbHNlO1xuXHRcdFx0bGFzdFRpbWUgPSBEYXRlLm5vdygpO1xuXHRcdFx0Zm4oKTtcblx0XHR9O1xuXHRcdHZhciBpZGxlQ2FsbGJhY2sgPSByZXF1ZXN0SWRsZUNhbGxiYWNrICYmIHJJQ1RpbWVvdXQgPiA0OSA/XG5cdFx0XHRmdW5jdGlvbigpe1xuXHRcdFx0XHRyZXF1ZXN0SWRsZUNhbGxiYWNrKHJ1biwge3RpbWVvdXQ6IHJJQ1RpbWVvdXR9KTtcblxuXHRcdFx0XHRpZihySUNUaW1lb3V0ICE9PSBsYXp5U2l6ZXNDZmcucmljVGltZW91dCl7XG5cdFx0XHRcdFx0cklDVGltZW91dCA9IGxhenlTaXplc0NmZy5yaWNUaW1lb3V0O1xuXHRcdFx0XHR9XG5cdFx0XHR9IDpcblx0XHRcdHJBRkl0KGZ1bmN0aW9uKCl7XG5cdFx0XHRcdHNldFRpbWVvdXQocnVuKTtcblx0XHRcdH0sIHRydWUpXG5cdFx0O1xuXG5cdFx0cmV0dXJuIGZ1bmN0aW9uKGlzUHJpb3JpdHkpe1xuXHRcdFx0dmFyIGRlbGF5O1xuXG5cdFx0XHRpZigoaXNQcmlvcml0eSA9IGlzUHJpb3JpdHkgPT09IHRydWUpKXtcblx0XHRcdFx0cklDVGltZW91dCA9IDMzO1xuXHRcdFx0fVxuXG5cdFx0XHRpZihydW5uaW5nKXtcblx0XHRcdFx0cmV0dXJuO1xuXHRcdFx0fVxuXG5cdFx0XHRydW5uaW5nID0gIHRydWU7XG5cblx0XHRcdGRlbGF5ID0gZ0RlbGF5IC0gKERhdGUubm93KCkgLSBsYXN0VGltZSk7XG5cblx0XHRcdGlmKGRlbGF5IDwgMCl7XG5cdFx0XHRcdGRlbGF5ID0gMDtcblx0XHRcdH1cblxuXHRcdFx0aWYoaXNQcmlvcml0eSB8fCBkZWxheSA8IDkpe1xuXHRcdFx0XHRpZGxlQ2FsbGJhY2soKTtcblx0XHRcdH0gZWxzZSB7XG5cdFx0XHRcdHNldFRpbWVvdXQoaWRsZUNhbGxiYWNrLCBkZWxheSk7XG5cdFx0XHR9XG5cdFx0fTtcblx0fTtcblxuXHQvL2Jhc2VkIG9uIGh0dHA6Ly9tb2Rlcm5qYXZhc2NyaXB0LmJsb2dzcG90LmRlLzIwMTMvMDgvYnVpbGRpbmctYmV0dGVyLWRlYm91bmNlLmh0bWxcblx0dmFyIGRlYm91bmNlID0gZnVuY3Rpb24oZnVuYykge1xuXHRcdHZhciB0aW1lb3V0LCB0aW1lc3RhbXA7XG5cdFx0dmFyIHdhaXQgPSA5OTtcblx0XHR2YXIgcnVuID0gZnVuY3Rpb24oKXtcblx0XHRcdHRpbWVvdXQgPSBudWxsO1xuXHRcdFx0ZnVuYygpO1xuXHRcdH07XG5cdFx0dmFyIGxhdGVyID0gZnVuY3Rpb24oKSB7XG5cdFx0XHR2YXIgbGFzdCA9IERhdGUubm93KCkgLSB0aW1lc3RhbXA7XG5cblx0XHRcdGlmIChsYXN0IDwgd2FpdCkge1xuXHRcdFx0XHRzZXRUaW1lb3V0KGxhdGVyLCB3YWl0IC0gbGFzdCk7XG5cdFx0XHR9IGVsc2Uge1xuXHRcdFx0XHQocmVxdWVzdElkbGVDYWxsYmFjayB8fCBydW4pKHJ1bik7XG5cdFx0XHR9XG5cdFx0fTtcblxuXHRcdHJldHVybiBmdW5jdGlvbigpIHtcblx0XHRcdHRpbWVzdGFtcCA9IERhdGUubm93KCk7XG5cblx0XHRcdGlmICghdGltZW91dCkge1xuXHRcdFx0XHR0aW1lb3V0ID0gc2V0VGltZW91dChsYXRlciwgd2FpdCk7XG5cdFx0XHR9XG5cdFx0fTtcblx0fTtcblxuXHR2YXIgbG9hZGVyID0gKGZ1bmN0aW9uKCl7XG5cdFx0dmFyIHByZWxvYWRFbGVtcywgaXNDb21wbGV0ZWQsIHJlc2V0UHJlbG9hZGluZ1RpbWVyLCBsb2FkTW9kZSwgc3RhcnRlZDtcblxuXHRcdHZhciBlTHZXLCBlbHZILCBlTHRvcCwgZUxsZWZ0LCBlTHJpZ2h0LCBlTGJvdHRvbSwgaXNCb2R5SGlkZGVuO1xuXG5cdFx0dmFyIHJlZ0ltZyA9IC9eaW1nJC9pO1xuXHRcdHZhciByZWdJZnJhbWUgPSAvXmlmcmFtZSQvaTtcblxuXHRcdHZhciBzdXBwb3J0U2Nyb2xsID0gKCdvbnNjcm9sbCcgaW4gd2luZG93KSAmJiAhKC8oZ2xlfGluZylib3QvLnRlc3QobmF2aWdhdG9yLnVzZXJBZ2VudCkpO1xuXG5cdFx0dmFyIHNocmlua0V4cGFuZCA9IDA7XG5cdFx0dmFyIGN1cnJlbnRFeHBhbmQgPSAwO1xuXG5cdFx0dmFyIGlzTG9hZGluZyA9IDA7XG5cdFx0dmFyIGxvd1J1bnMgPSAtMTtcblxuXHRcdHZhciByZXNldFByZWxvYWRpbmcgPSBmdW5jdGlvbihlKXtcblx0XHRcdGlzTG9hZGluZy0tO1xuXHRcdFx0aWYoIWUgfHwgaXNMb2FkaW5nIDwgMCB8fCAhZS50YXJnZXQpe1xuXHRcdFx0XHRpc0xvYWRpbmcgPSAwO1xuXHRcdFx0fVxuXHRcdH07XG5cblx0XHR2YXIgaXNWaXNpYmxlID0gZnVuY3Rpb24gKGVsZW0pIHtcblx0XHRcdGlmIChpc0JvZHlIaWRkZW4gPT0gbnVsbCkge1xuXHRcdFx0XHRpc0JvZHlIaWRkZW4gPSBnZXRDU1MoZG9jdW1lbnQuYm9keSwgJ3Zpc2liaWxpdHknKSA9PSAnaGlkZGVuJztcblx0XHRcdH1cblxuXHRcdFx0cmV0dXJuIGlzQm9keUhpZGRlbiB8fCAhKGdldENTUyhlbGVtLnBhcmVudE5vZGUsICd2aXNpYmlsaXR5JykgPT0gJ2hpZGRlbicgJiYgZ2V0Q1NTKGVsZW0sICd2aXNpYmlsaXR5JykgPT0gJ2hpZGRlbicpO1xuXHRcdH07XG5cblx0XHR2YXIgaXNOZXN0ZWRWaXNpYmxlID0gZnVuY3Rpb24oZWxlbSwgZWxlbUV4cGFuZCl7XG5cdFx0XHR2YXIgb3V0ZXJSZWN0O1xuXHRcdFx0dmFyIHBhcmVudCA9IGVsZW07XG5cdFx0XHR2YXIgdmlzaWJsZSA9IGlzVmlzaWJsZShlbGVtKTtcblxuXHRcdFx0ZUx0b3AgLT0gZWxlbUV4cGFuZDtcblx0XHRcdGVMYm90dG9tICs9IGVsZW1FeHBhbmQ7XG5cdFx0XHRlTGxlZnQgLT0gZWxlbUV4cGFuZDtcblx0XHRcdGVMcmlnaHQgKz0gZWxlbUV4cGFuZDtcblxuXHRcdFx0d2hpbGUodmlzaWJsZSAmJiAocGFyZW50ID0gcGFyZW50Lm9mZnNldFBhcmVudCkgJiYgcGFyZW50ICE9IGRvY3VtZW50LmJvZHkgJiYgcGFyZW50ICE9IGRvY0VsZW0pe1xuXHRcdFx0XHR2aXNpYmxlID0gKChnZXRDU1MocGFyZW50LCAnb3BhY2l0eScpIHx8IDEpID4gMCk7XG5cblx0XHRcdFx0aWYodmlzaWJsZSAmJiBnZXRDU1MocGFyZW50LCAnb3ZlcmZsb3cnKSAhPSAndmlzaWJsZScpe1xuXHRcdFx0XHRcdG91dGVyUmVjdCA9IHBhcmVudC5nZXRCb3VuZGluZ0NsaWVudFJlY3QoKTtcblx0XHRcdFx0XHR2aXNpYmxlID0gZUxyaWdodCA+IG91dGVyUmVjdC5sZWZ0ICYmXG5cdFx0XHRcdFx0XHRlTGxlZnQgPCBvdXRlclJlY3QucmlnaHQgJiZcblx0XHRcdFx0XHRcdGVMYm90dG9tID4gb3V0ZXJSZWN0LnRvcCAtIDEgJiZcblx0XHRcdFx0XHRcdGVMdG9wIDwgb3V0ZXJSZWN0LmJvdHRvbSArIDFcblx0XHRcdFx0XHQ7XG5cdFx0XHRcdH1cblx0XHRcdH1cblxuXHRcdFx0cmV0dXJuIHZpc2libGU7XG5cdFx0fTtcblxuXHRcdHZhciBjaGVja0VsZW1lbnRzID0gZnVuY3Rpb24oKSB7XG5cdFx0XHR2YXIgZUxsZW4sIGksIHJlY3QsIGF1dG9Mb2FkRWxlbSwgbG9hZGVkU29tZXRoaW5nLCBlbGVtRXhwYW5kLCBlbGVtTmVnYXRpdmVFeHBhbmQsIGVsZW1FeHBhbmRWYWwsXG5cdFx0XHRcdGJlZm9yZUV4cGFuZFZhbCwgZGVmYXVsdEV4cGFuZCwgcHJlbG9hZEV4cGFuZCwgaEZhYztcblx0XHRcdHZhciBsYXp5bG9hZEVsZW1zID0gbGF6eXNpemVzLmVsZW1lbnRzO1xuXG5cdFx0XHRpZigobG9hZE1vZGUgPSBsYXp5U2l6ZXNDZmcubG9hZE1vZGUpICYmIGlzTG9hZGluZyA8IDggJiYgKGVMbGVuID0gbGF6eWxvYWRFbGVtcy5sZW5ndGgpKXtcblxuXHRcdFx0XHRpID0gMDtcblxuXHRcdFx0XHRsb3dSdW5zKys7XG5cblx0XHRcdFx0Zm9yKDsgaSA8IGVMbGVuOyBpKyspe1xuXG5cdFx0XHRcdFx0aWYoIWxhenlsb2FkRWxlbXNbaV0gfHwgbGF6eWxvYWRFbGVtc1tpXS5fbGF6eVJhY2Upe2NvbnRpbnVlO31cblxuXHRcdFx0XHRcdGlmKCFzdXBwb3J0U2Nyb2xsIHx8IChsYXp5c2l6ZXMucHJlbWF0dXJlVW52ZWlsICYmIGxhenlzaXplcy5wcmVtYXR1cmVVbnZlaWwobGF6eWxvYWRFbGVtc1tpXSkpKXt1bnZlaWxFbGVtZW50KGxhenlsb2FkRWxlbXNbaV0pO2NvbnRpbnVlO31cblxuXHRcdFx0XHRcdGlmKCEoZWxlbUV4cGFuZFZhbCA9IGxhenlsb2FkRWxlbXNbaV1bX2dldEF0dHJpYnV0ZV0oJ2RhdGEtZXhwYW5kJykpIHx8ICEoZWxlbUV4cGFuZCA9IGVsZW1FeHBhbmRWYWwgKiAxKSl7XG5cdFx0XHRcdFx0XHRlbGVtRXhwYW5kID0gY3VycmVudEV4cGFuZDtcblx0XHRcdFx0XHR9XG5cblx0XHRcdFx0XHRpZiAoIWRlZmF1bHRFeHBhbmQpIHtcblx0XHRcdFx0XHRcdGRlZmF1bHRFeHBhbmQgPSAoIWxhenlTaXplc0NmZy5leHBhbmQgfHwgbGF6eVNpemVzQ2ZnLmV4cGFuZCA8IDEpID9cblx0XHRcdFx0XHRcdFx0ZG9jRWxlbS5jbGllbnRIZWlnaHQgPiA1MDAgJiYgZG9jRWxlbS5jbGllbnRXaWR0aCA+IDUwMCA/IDUwMCA6IDM3MCA6XG5cdFx0XHRcdFx0XHRcdGxhenlTaXplc0NmZy5leHBhbmQ7XG5cblx0XHRcdFx0XHRcdGxhenlzaXplcy5fZGVmRXggPSBkZWZhdWx0RXhwYW5kO1xuXG5cdFx0XHRcdFx0XHRwcmVsb2FkRXhwYW5kID0gZGVmYXVsdEV4cGFuZCAqIGxhenlTaXplc0NmZy5leHBGYWN0b3I7XG5cdFx0XHRcdFx0XHRoRmFjID0gbGF6eVNpemVzQ2ZnLmhGYWM7XG5cdFx0XHRcdFx0XHRpc0JvZHlIaWRkZW4gPSBudWxsO1xuXG5cdFx0XHRcdFx0XHRpZihjdXJyZW50RXhwYW5kIDwgcHJlbG9hZEV4cGFuZCAmJiBpc0xvYWRpbmcgPCAxICYmIGxvd1J1bnMgPiAyICYmIGxvYWRNb2RlID4gMiAmJiAhZG9jdW1lbnQuaGlkZGVuKXtcblx0XHRcdFx0XHRcdFx0Y3VycmVudEV4cGFuZCA9IHByZWxvYWRFeHBhbmQ7XG5cdFx0XHRcdFx0XHRcdGxvd1J1bnMgPSAwO1xuXHRcdFx0XHRcdFx0fSBlbHNlIGlmKGxvYWRNb2RlID4gMSAmJiBsb3dSdW5zID4gMSAmJiBpc0xvYWRpbmcgPCA2KXtcblx0XHRcdFx0XHRcdFx0Y3VycmVudEV4cGFuZCA9IGRlZmF1bHRFeHBhbmQ7XG5cdFx0XHRcdFx0XHR9IGVsc2Uge1xuXHRcdFx0XHRcdFx0XHRjdXJyZW50RXhwYW5kID0gc2hyaW5rRXhwYW5kO1xuXHRcdFx0XHRcdFx0fVxuXHRcdFx0XHRcdH1cblxuXHRcdFx0XHRcdGlmKGJlZm9yZUV4cGFuZFZhbCAhPT0gZWxlbUV4cGFuZCl7XG5cdFx0XHRcdFx0XHRlTHZXID0gaW5uZXJXaWR0aCArIChlbGVtRXhwYW5kICogaEZhYyk7XG5cdFx0XHRcdFx0XHRlbHZIID0gaW5uZXJIZWlnaHQgKyBlbGVtRXhwYW5kO1xuXHRcdFx0XHRcdFx0ZWxlbU5lZ2F0aXZlRXhwYW5kID0gZWxlbUV4cGFuZCAqIC0xO1xuXHRcdFx0XHRcdFx0YmVmb3JlRXhwYW5kVmFsID0gZWxlbUV4cGFuZDtcblx0XHRcdFx0XHR9XG5cblx0XHRcdFx0XHRyZWN0ID0gbGF6eWxvYWRFbGVtc1tpXS5nZXRCb3VuZGluZ0NsaWVudFJlY3QoKTtcblxuXHRcdFx0XHRcdGlmICgoZUxib3R0b20gPSByZWN0LmJvdHRvbSkgPj0gZWxlbU5lZ2F0aXZlRXhwYW5kICYmXG5cdFx0XHRcdFx0XHQoZUx0b3AgPSByZWN0LnRvcCkgPD0gZWx2SCAmJlxuXHRcdFx0XHRcdFx0KGVMcmlnaHQgPSByZWN0LnJpZ2h0KSA+PSBlbGVtTmVnYXRpdmVFeHBhbmQgKiBoRmFjICYmXG5cdFx0XHRcdFx0XHQoZUxsZWZ0ID0gcmVjdC5sZWZ0KSA8PSBlTHZXICYmXG5cdFx0XHRcdFx0XHQoZUxib3R0b20gfHwgZUxyaWdodCB8fCBlTGxlZnQgfHwgZUx0b3ApICYmXG5cdFx0XHRcdFx0XHQobGF6eVNpemVzQ2ZnLmxvYWRIaWRkZW4gfHwgaXNWaXNpYmxlKGxhenlsb2FkRWxlbXNbaV0pKSAmJlxuXHRcdFx0XHRcdFx0KChpc0NvbXBsZXRlZCAmJiBpc0xvYWRpbmcgPCAzICYmICFlbGVtRXhwYW5kVmFsICYmIChsb2FkTW9kZSA8IDMgfHwgbG93UnVucyA8IDQpKSB8fCBpc05lc3RlZFZpc2libGUobGF6eWxvYWRFbGVtc1tpXSwgZWxlbUV4cGFuZCkpKXtcblx0XHRcdFx0XHRcdHVudmVpbEVsZW1lbnQobGF6eWxvYWRFbGVtc1tpXSk7XG5cdFx0XHRcdFx0XHRsb2FkZWRTb21ldGhpbmcgPSB0cnVlO1xuXHRcdFx0XHRcdFx0aWYoaXNMb2FkaW5nID4gOSl7YnJlYWs7fVxuXHRcdFx0XHRcdH0gZWxzZSBpZighbG9hZGVkU29tZXRoaW5nICYmIGlzQ29tcGxldGVkICYmICFhdXRvTG9hZEVsZW0gJiZcblx0XHRcdFx0XHRcdGlzTG9hZGluZyA8IDQgJiYgbG93UnVucyA8IDQgJiYgbG9hZE1vZGUgPiAyICYmXG5cdFx0XHRcdFx0XHQocHJlbG9hZEVsZW1zWzBdIHx8IGxhenlTaXplc0NmZy5wcmVsb2FkQWZ0ZXJMb2FkKSAmJlxuXHRcdFx0XHRcdFx0KHByZWxvYWRFbGVtc1swXSB8fCAoIWVsZW1FeHBhbmRWYWwgJiYgKChlTGJvdHRvbSB8fCBlTHJpZ2h0IHx8IGVMbGVmdCB8fCBlTHRvcCkgfHwgbGF6eWxvYWRFbGVtc1tpXVtfZ2V0QXR0cmlidXRlXShsYXp5U2l6ZXNDZmcuc2l6ZXNBdHRyKSAhPSAnYXV0bycpKSkpe1xuXHRcdFx0XHRcdFx0YXV0b0xvYWRFbGVtID0gcHJlbG9hZEVsZW1zWzBdIHx8IGxhenlsb2FkRWxlbXNbaV07XG5cdFx0XHRcdFx0fVxuXHRcdFx0XHR9XG5cblx0XHRcdFx0aWYoYXV0b0xvYWRFbGVtICYmICFsb2FkZWRTb21ldGhpbmcpe1xuXHRcdFx0XHRcdHVudmVpbEVsZW1lbnQoYXV0b0xvYWRFbGVtKTtcblx0XHRcdFx0fVxuXHRcdFx0fVxuXHRcdH07XG5cblx0XHR2YXIgdGhyb3R0bGVkQ2hlY2tFbGVtZW50cyA9IHRocm90dGxlKGNoZWNrRWxlbWVudHMpO1xuXG5cdFx0dmFyIHN3aXRjaExvYWRpbmdDbGFzcyA9IGZ1bmN0aW9uKGUpe1xuXHRcdFx0dmFyIGVsZW0gPSBlLnRhcmdldDtcblxuXHRcdFx0aWYgKGVsZW0uX2xhenlDYWNoZSkge1xuXHRcdFx0XHRkZWxldGUgZWxlbS5fbGF6eUNhY2hlO1xuXHRcdFx0XHRyZXR1cm47XG5cdFx0XHR9XG5cblx0XHRcdHJlc2V0UHJlbG9hZGluZyhlKTtcblx0XHRcdGFkZENsYXNzKGVsZW0sIGxhenlTaXplc0NmZy5sb2FkZWRDbGFzcyk7XG5cdFx0XHRyZW1vdmVDbGFzcyhlbGVtLCBsYXp5U2l6ZXNDZmcubG9hZGluZ0NsYXNzKTtcblx0XHRcdGFkZFJlbW92ZUxvYWRFdmVudHMoZWxlbSwgcmFmU3dpdGNoTG9hZGluZ0NsYXNzKTtcblx0XHRcdHRyaWdnZXJFdmVudChlbGVtLCAnbGF6eWxvYWRlZCcpO1xuXHRcdH07XG5cdFx0dmFyIHJhZmVkU3dpdGNoTG9hZGluZ0NsYXNzID0gckFGSXQoc3dpdGNoTG9hZGluZ0NsYXNzKTtcblx0XHR2YXIgcmFmU3dpdGNoTG9hZGluZ0NsYXNzID0gZnVuY3Rpb24oZSl7XG5cdFx0XHRyYWZlZFN3aXRjaExvYWRpbmdDbGFzcyh7dGFyZ2V0OiBlLnRhcmdldH0pO1xuXHRcdH07XG5cblx0XHR2YXIgY2hhbmdlSWZyYW1lU3JjID0gZnVuY3Rpb24oZWxlbSwgc3JjKXtcblx0XHRcdHZhciBsb2FkTW9kZSA9IGVsZW0uZ2V0QXR0cmlidXRlKCdkYXRhLWxvYWQtbW9kZScpIHx8IGxhenlTaXplc0NmZy5pZnJhbWVMb2FkTW9kZTtcblxuXHRcdFx0Ly8gbG9hZE1vZGUgY2FuIGJlIGFsc28gYSBzdHJpbmchXG5cdFx0XHRpZiAobG9hZE1vZGUgPT0gMCkge1xuXHRcdFx0XHRlbGVtLmNvbnRlbnRXaW5kb3cubG9jYXRpb24ucmVwbGFjZShzcmMpO1xuXHRcdFx0fSBlbHNlIGlmIChsb2FkTW9kZSA9PSAxKSB7XG5cdFx0XHRcdGVsZW0uc3JjID0gc3JjO1xuXHRcdFx0fVxuXHRcdH07XG5cblx0XHR2YXIgaGFuZGxlU291cmNlcyA9IGZ1bmN0aW9uKHNvdXJjZSl7XG5cdFx0XHR2YXIgY3VzdG9tTWVkaWE7XG5cblx0XHRcdHZhciBzb3VyY2VTcmNzZXQgPSBzb3VyY2VbX2dldEF0dHJpYnV0ZV0obGF6eVNpemVzQ2ZnLnNyY3NldEF0dHIpO1xuXG5cdFx0XHRpZiggKGN1c3RvbU1lZGlhID0gbGF6eVNpemVzQ2ZnLmN1c3RvbU1lZGlhW3NvdXJjZVtfZ2V0QXR0cmlidXRlXSgnZGF0YS1tZWRpYScpIHx8IHNvdXJjZVtfZ2V0QXR0cmlidXRlXSgnbWVkaWEnKV0pICl7XG5cdFx0XHRcdHNvdXJjZS5zZXRBdHRyaWJ1dGUoJ21lZGlhJywgY3VzdG9tTWVkaWEpO1xuXHRcdFx0fVxuXG5cdFx0XHRpZihzb3VyY2VTcmNzZXQpe1xuXHRcdFx0XHRzb3VyY2Uuc2V0QXR0cmlidXRlKCdzcmNzZXQnLCBzb3VyY2VTcmNzZXQpO1xuXHRcdFx0fVxuXHRcdH07XG5cblx0XHR2YXIgbGF6eVVudmVpbCA9IHJBRkl0KGZ1bmN0aW9uIChlbGVtLCBkZXRhaWwsIGlzQXV0bywgc2l6ZXMsIGlzSW1nKXtcblx0XHRcdHZhciBzcmMsIHNyY3NldCwgcGFyZW50LCBpc1BpY3R1cmUsIGV2ZW50LCBmaXJlc0xvYWQ7XG5cblx0XHRcdGlmKCEoZXZlbnQgPSB0cmlnZ2VyRXZlbnQoZWxlbSwgJ2xhenliZWZvcmV1bnZlaWwnLCBkZXRhaWwpKS5kZWZhdWx0UHJldmVudGVkKXtcblxuXHRcdFx0XHRpZihzaXplcyl7XG5cdFx0XHRcdFx0aWYoaXNBdXRvKXtcblx0XHRcdFx0XHRcdGFkZENsYXNzKGVsZW0sIGxhenlTaXplc0NmZy5hdXRvc2l6ZXNDbGFzcyk7XG5cdFx0XHRcdFx0fSBlbHNlIHtcblx0XHRcdFx0XHRcdGVsZW0uc2V0QXR0cmlidXRlKCdzaXplcycsIHNpemVzKTtcblx0XHRcdFx0XHR9XG5cdFx0XHRcdH1cblxuXHRcdFx0XHRzcmNzZXQgPSBlbGVtW19nZXRBdHRyaWJ1dGVdKGxhenlTaXplc0NmZy5zcmNzZXRBdHRyKTtcblx0XHRcdFx0c3JjID0gZWxlbVtfZ2V0QXR0cmlidXRlXShsYXp5U2l6ZXNDZmcuc3JjQXR0cik7XG5cblx0XHRcdFx0aWYoaXNJbWcpIHtcblx0XHRcdFx0XHRwYXJlbnQgPSBlbGVtLnBhcmVudE5vZGU7XG5cdFx0XHRcdFx0aXNQaWN0dXJlID0gcGFyZW50ICYmIHJlZ1BpY3R1cmUudGVzdChwYXJlbnQubm9kZU5hbWUgfHwgJycpO1xuXHRcdFx0XHR9XG5cblx0XHRcdFx0ZmlyZXNMb2FkID0gZGV0YWlsLmZpcmVzTG9hZCB8fCAoKCdzcmMnIGluIGVsZW0pICYmIChzcmNzZXQgfHwgc3JjIHx8IGlzUGljdHVyZSkpO1xuXG5cdFx0XHRcdGV2ZW50ID0ge3RhcmdldDogZWxlbX07XG5cblx0XHRcdFx0YWRkQ2xhc3MoZWxlbSwgbGF6eVNpemVzQ2ZnLmxvYWRpbmdDbGFzcyk7XG5cblx0XHRcdFx0aWYoZmlyZXNMb2FkKXtcblx0XHRcdFx0XHRjbGVhclRpbWVvdXQocmVzZXRQcmVsb2FkaW5nVGltZXIpO1xuXHRcdFx0XHRcdHJlc2V0UHJlbG9hZGluZ1RpbWVyID0gc2V0VGltZW91dChyZXNldFByZWxvYWRpbmcsIDI1MDApO1xuXHRcdFx0XHRcdGFkZFJlbW92ZUxvYWRFdmVudHMoZWxlbSwgcmFmU3dpdGNoTG9hZGluZ0NsYXNzLCB0cnVlKTtcblx0XHRcdFx0fVxuXG5cdFx0XHRcdGlmKGlzUGljdHVyZSl7XG5cdFx0XHRcdFx0Zm9yRWFjaC5jYWxsKHBhcmVudC5nZXRFbGVtZW50c0J5VGFnTmFtZSgnc291cmNlJyksIGhhbmRsZVNvdXJjZXMpO1xuXHRcdFx0XHR9XG5cblx0XHRcdFx0aWYoc3Jjc2V0KXtcblx0XHRcdFx0XHRlbGVtLnNldEF0dHJpYnV0ZSgnc3Jjc2V0Jywgc3Jjc2V0KTtcblx0XHRcdFx0fSBlbHNlIGlmKHNyYyAmJiAhaXNQaWN0dXJlKXtcblx0XHRcdFx0XHRpZihyZWdJZnJhbWUudGVzdChlbGVtLm5vZGVOYW1lKSl7XG5cdFx0XHRcdFx0XHRjaGFuZ2VJZnJhbWVTcmMoZWxlbSwgc3JjKTtcblx0XHRcdFx0XHR9IGVsc2Uge1xuXHRcdFx0XHRcdFx0ZWxlbS5zcmMgPSBzcmM7XG5cdFx0XHRcdFx0fVxuXHRcdFx0XHR9XG5cblx0XHRcdFx0aWYoaXNJbWcgJiYgKHNyY3NldCB8fCBpc1BpY3R1cmUpKXtcblx0XHRcdFx0XHR1cGRhdGVQb2x5ZmlsbChlbGVtLCB7c3JjOiBzcmN9KTtcblx0XHRcdFx0fVxuXHRcdFx0fVxuXG5cdFx0XHRpZihlbGVtLl9sYXp5UmFjZSl7XG5cdFx0XHRcdGRlbGV0ZSBlbGVtLl9sYXp5UmFjZTtcblx0XHRcdH1cblx0XHRcdHJlbW92ZUNsYXNzKGVsZW0sIGxhenlTaXplc0NmZy5sYXp5Q2xhc3MpO1xuXG5cdFx0XHRyQUYoZnVuY3Rpb24oKXtcblx0XHRcdFx0Ly8gUGFydCBvZiB0aGlzIGNhbiBiZSByZW1vdmVkIGFzIHNvb24gYXMgdGhpcyBmaXggaXMgb2xkZXI6IGh0dHBzOi8vYnVncy5jaHJvbWl1bS5vcmcvcC9jaHJvbWl1bS9pc3N1ZXMvZGV0YWlsP2lkPTc3MzEgKDIwMTUpXG5cdFx0XHRcdHZhciBpc0xvYWRlZCA9IGVsZW0uY29tcGxldGUgJiYgZWxlbS5uYXR1cmFsV2lkdGggPiAxO1xuXG5cdFx0XHRcdGlmKCAhZmlyZXNMb2FkIHx8IGlzTG9hZGVkKXtcblx0XHRcdFx0XHRpZiAoaXNMb2FkZWQpIHtcblx0XHRcdFx0XHRcdGFkZENsYXNzKGVsZW0sIGxhenlTaXplc0NmZy5mYXN0TG9hZGVkQ2xhc3MpO1xuXHRcdFx0XHRcdH1cblx0XHRcdFx0XHRzd2l0Y2hMb2FkaW5nQ2xhc3MoZXZlbnQpO1xuXHRcdFx0XHRcdGVsZW0uX2xhenlDYWNoZSA9IHRydWU7XG5cdFx0XHRcdFx0c2V0VGltZW91dChmdW5jdGlvbigpe1xuXHRcdFx0XHRcdFx0aWYgKCdfbGF6eUNhY2hlJyBpbiBlbGVtKSB7XG5cdFx0XHRcdFx0XHRcdGRlbGV0ZSBlbGVtLl9sYXp5Q2FjaGU7XG5cdFx0XHRcdFx0XHR9XG5cdFx0XHRcdFx0fSwgOSk7XG5cdFx0XHRcdH1cblx0XHRcdFx0aWYgKGVsZW0ubG9hZGluZyA9PSAnbGF6eScpIHtcblx0XHRcdFx0XHRpc0xvYWRpbmctLTtcblx0XHRcdFx0fVxuXHRcdFx0fSwgdHJ1ZSk7XG5cdFx0fSk7XG5cblx0XHQvKipcblx0XHQgKlxuXHRcdCAqIEBwYXJhbSBlbGVtIHsgRWxlbWVudCB9XG5cdFx0ICovXG5cdFx0dmFyIHVudmVpbEVsZW1lbnQgPSBmdW5jdGlvbiAoZWxlbSl7XG5cdFx0XHRpZiAoZWxlbS5fbGF6eVJhY2UpIHtyZXR1cm47fVxuXHRcdFx0dmFyIGRldGFpbDtcblxuXHRcdFx0dmFyIGlzSW1nID0gcmVnSW1nLnRlc3QoZWxlbS5ub2RlTmFtZSk7XG5cblx0XHRcdC8vYWxsb3cgdXNpbmcgc2l6ZXM9XCJhdXRvXCIsIGJ1dCBkb24ndCB1c2UuIGl0J3MgaW52YWxpZC4gVXNlIGRhdGEtc2l6ZXM9XCJhdXRvXCIgb3IgYSB2YWxpZCB2YWx1ZSBmb3Igc2l6ZXMgaW5zdGVhZCAoaS5lLjogc2l6ZXM9XCI4MHZ3XCIpXG5cdFx0XHR2YXIgc2l6ZXMgPSBpc0ltZyAmJiAoZWxlbVtfZ2V0QXR0cmlidXRlXShsYXp5U2l6ZXNDZmcuc2l6ZXNBdHRyKSB8fCBlbGVtW19nZXRBdHRyaWJ1dGVdKCdzaXplcycpKTtcblx0XHRcdHZhciBpc0F1dG8gPSBzaXplcyA9PSAnYXV0byc7XG5cblx0XHRcdGlmKCAoaXNBdXRvIHx8ICFpc0NvbXBsZXRlZCkgJiYgaXNJbWcgJiYgKGVsZW1bX2dldEF0dHJpYnV0ZV0oJ3NyYycpIHx8IGVsZW0uc3Jjc2V0KSAmJiAhZWxlbS5jb21wbGV0ZSAmJiAhaGFzQ2xhc3MoZWxlbSwgbGF6eVNpemVzQ2ZnLmVycm9yQ2xhc3MpICYmIGhhc0NsYXNzKGVsZW0sIGxhenlTaXplc0NmZy5sYXp5Q2xhc3MpKXtyZXR1cm47fVxuXG5cdFx0XHRkZXRhaWwgPSB0cmlnZ2VyRXZlbnQoZWxlbSwgJ2xhenl1bnZlaWxyZWFkJykuZGV0YWlsO1xuXG5cdFx0XHRpZihpc0F1dG8pe1xuXHRcdFx0XHQgYXV0b1NpemVyLnVwZGF0ZUVsZW0oZWxlbSwgdHJ1ZSwgZWxlbS5vZmZzZXRXaWR0aCk7XG5cdFx0XHR9XG5cblx0XHRcdGVsZW0uX2xhenlSYWNlID0gdHJ1ZTtcblx0XHRcdGlzTG9hZGluZysrO1xuXG5cdFx0XHRsYXp5VW52ZWlsKGVsZW0sIGRldGFpbCwgaXNBdXRvLCBzaXplcywgaXNJbWcpO1xuXHRcdH07XG5cblx0XHR2YXIgYWZ0ZXJTY3JvbGwgPSBkZWJvdW5jZShmdW5jdGlvbigpe1xuXHRcdFx0bGF6eVNpemVzQ2ZnLmxvYWRNb2RlID0gMztcblx0XHRcdHRocm90dGxlZENoZWNrRWxlbWVudHMoKTtcblx0XHR9KTtcblxuXHRcdHZhciBhbHRMb2FkbW9kZVNjcm9sbExpc3RuZXIgPSBmdW5jdGlvbigpe1xuXHRcdFx0aWYobGF6eVNpemVzQ2ZnLmxvYWRNb2RlID09IDMpe1xuXHRcdFx0XHRsYXp5U2l6ZXNDZmcubG9hZE1vZGUgPSAyO1xuXHRcdFx0fVxuXHRcdFx0YWZ0ZXJTY3JvbGwoKTtcblx0XHR9O1xuXG5cdFx0dmFyIG9ubG9hZCA9IGZ1bmN0aW9uKCl7XG5cdFx0XHRpZihpc0NvbXBsZXRlZCl7cmV0dXJuO31cblx0XHRcdGlmKERhdGUubm93KCkgLSBzdGFydGVkIDwgOTk5KXtcblx0XHRcdFx0c2V0VGltZW91dChvbmxvYWQsIDk5OSk7XG5cdFx0XHRcdHJldHVybjtcblx0XHRcdH1cblxuXG5cdFx0XHRpc0NvbXBsZXRlZCA9IHRydWU7XG5cblx0XHRcdGxhenlTaXplc0NmZy5sb2FkTW9kZSA9IDM7XG5cblx0XHRcdHRocm90dGxlZENoZWNrRWxlbWVudHMoKTtcblxuXHRcdFx0YWRkRXZlbnRMaXN0ZW5lcignc2Nyb2xsJywgYWx0TG9hZG1vZGVTY3JvbGxMaXN0bmVyLCB0cnVlKTtcblx0XHR9O1xuXG5cdFx0cmV0dXJuIHtcblx0XHRcdF86IGZ1bmN0aW9uKCl7XG5cdFx0XHRcdHN0YXJ0ZWQgPSBEYXRlLm5vdygpO1xuXG5cdFx0XHRcdGxhenlzaXplcy5lbGVtZW50cyA9IGRvY3VtZW50LmdldEVsZW1lbnRzQnlDbGFzc05hbWUobGF6eVNpemVzQ2ZnLmxhenlDbGFzcyk7XG5cdFx0XHRcdHByZWxvYWRFbGVtcyA9IGRvY3VtZW50LmdldEVsZW1lbnRzQnlDbGFzc05hbWUobGF6eVNpemVzQ2ZnLmxhenlDbGFzcyArICcgJyArIGxhenlTaXplc0NmZy5wcmVsb2FkQ2xhc3MpO1xuXG5cdFx0XHRcdGFkZEV2ZW50TGlzdGVuZXIoJ3Njcm9sbCcsIHRocm90dGxlZENoZWNrRWxlbWVudHMsIHRydWUpO1xuXG5cdFx0XHRcdGFkZEV2ZW50TGlzdGVuZXIoJ3Jlc2l6ZScsIHRocm90dGxlZENoZWNrRWxlbWVudHMsIHRydWUpO1xuXG5cdFx0XHRcdGFkZEV2ZW50TGlzdGVuZXIoJ3BhZ2VzaG93JywgZnVuY3Rpb24gKGUpIHtcblx0XHRcdFx0XHRpZiAoZS5wZXJzaXN0ZWQpIHtcblx0XHRcdFx0XHRcdHZhciBsb2FkaW5nRWxlbWVudHMgPSBkb2N1bWVudC5xdWVyeVNlbGVjdG9yQWxsKCcuJyArIGxhenlTaXplc0NmZy5sb2FkaW5nQ2xhc3MpO1xuXG5cdFx0XHRcdFx0XHRpZiAobG9hZGluZ0VsZW1lbnRzLmxlbmd0aCAmJiBsb2FkaW5nRWxlbWVudHMuZm9yRWFjaCkge1xuXHRcdFx0XHRcdFx0XHRyZXF1ZXN0QW5pbWF0aW9uRnJhbWUoZnVuY3Rpb24gKCkge1xuXHRcdFx0XHRcdFx0XHRcdGxvYWRpbmdFbGVtZW50cy5mb3JFYWNoKCBmdW5jdGlvbiAoaW1nKSB7XG5cdFx0XHRcdFx0XHRcdFx0XHRpZiAoaW1nLmNvbXBsZXRlKSB7XG5cdFx0XHRcdFx0XHRcdFx0XHRcdHVudmVpbEVsZW1lbnQoaW1nKTtcblx0XHRcdFx0XHRcdFx0XHRcdH1cblx0XHRcdFx0XHRcdFx0XHR9KTtcblx0XHRcdFx0XHRcdFx0fSk7XG5cdFx0XHRcdFx0XHR9XG5cdFx0XHRcdFx0fVxuXHRcdFx0XHR9KTtcblxuXHRcdFx0XHRpZih3aW5kb3cuTXV0YXRpb25PYnNlcnZlcil7XG5cdFx0XHRcdFx0bmV3IE11dGF0aW9uT2JzZXJ2ZXIoIHRocm90dGxlZENoZWNrRWxlbWVudHMgKS5vYnNlcnZlKCBkb2NFbGVtLCB7Y2hpbGRMaXN0OiB0cnVlLCBzdWJ0cmVlOiB0cnVlLCBhdHRyaWJ1dGVzOiB0cnVlfSApO1xuXHRcdFx0XHR9IGVsc2Uge1xuXHRcdFx0XHRcdGRvY0VsZW1bX2FkZEV2ZW50TGlzdGVuZXJdKCdET01Ob2RlSW5zZXJ0ZWQnLCB0aHJvdHRsZWRDaGVja0VsZW1lbnRzLCB0cnVlKTtcblx0XHRcdFx0XHRkb2NFbGVtW19hZGRFdmVudExpc3RlbmVyXSgnRE9NQXR0ck1vZGlmaWVkJywgdGhyb3R0bGVkQ2hlY2tFbGVtZW50cywgdHJ1ZSk7XG5cdFx0XHRcdFx0c2V0SW50ZXJ2YWwodGhyb3R0bGVkQ2hlY2tFbGVtZW50cywgOTk5KTtcblx0XHRcdFx0fVxuXG5cdFx0XHRcdGFkZEV2ZW50TGlzdGVuZXIoJ2hhc2hjaGFuZ2UnLCB0aHJvdHRsZWRDaGVja0VsZW1lbnRzLCB0cnVlKTtcblxuXHRcdFx0XHQvLywgJ2Z1bGxzY3JlZW5jaGFuZ2UnXG5cdFx0XHRcdFsnZm9jdXMnLCAnbW91c2VvdmVyJywgJ2NsaWNrJywgJ2xvYWQnLCAndHJhbnNpdGlvbmVuZCcsICdhbmltYXRpb25lbmQnXS5mb3JFYWNoKGZ1bmN0aW9uKG5hbWUpe1xuXHRcdFx0XHRcdGRvY3VtZW50W19hZGRFdmVudExpc3RlbmVyXShuYW1lLCB0aHJvdHRsZWRDaGVja0VsZW1lbnRzLCB0cnVlKTtcblx0XHRcdFx0fSk7XG5cblx0XHRcdFx0aWYoKC9kJHxeYy8udGVzdChkb2N1bWVudC5yZWFkeVN0YXRlKSkpe1xuXHRcdFx0XHRcdG9ubG9hZCgpO1xuXHRcdFx0XHR9IGVsc2Uge1xuXHRcdFx0XHRcdGFkZEV2ZW50TGlzdGVuZXIoJ2xvYWQnLCBvbmxvYWQpO1xuXHRcdFx0XHRcdGRvY3VtZW50W19hZGRFdmVudExpc3RlbmVyXSgnRE9NQ29udGVudExvYWRlZCcsIHRocm90dGxlZENoZWNrRWxlbWVudHMpO1xuXHRcdFx0XHRcdHNldFRpbWVvdXQob25sb2FkLCAyMDAwMCk7XG5cdFx0XHRcdH1cblxuXHRcdFx0XHRpZihsYXp5c2l6ZXMuZWxlbWVudHMubGVuZ3RoKXtcblx0XHRcdFx0XHRjaGVja0VsZW1lbnRzKCk7XG5cdFx0XHRcdFx0ckFGLl9sc0ZsdXNoKCk7XG5cdFx0XHRcdH0gZWxzZSB7XG5cdFx0XHRcdFx0dGhyb3R0bGVkQ2hlY2tFbGVtZW50cygpO1xuXHRcdFx0XHR9XG5cdFx0XHR9LFxuXHRcdFx0Y2hlY2tFbGVtczogdGhyb3R0bGVkQ2hlY2tFbGVtZW50cyxcblx0XHRcdHVudmVpbDogdW52ZWlsRWxlbWVudCxcblx0XHRcdF9hTFNMOiBhbHRMb2FkbW9kZVNjcm9sbExpc3RuZXIsXG5cdFx0fTtcblx0fSkoKTtcblxuXG5cdHZhciBhdXRvU2l6ZXIgPSAoZnVuY3Rpb24oKXtcblx0XHR2YXIgYXV0b3NpemVzRWxlbXM7XG5cblx0XHR2YXIgc2l6ZUVsZW1lbnQgPSByQUZJdChmdW5jdGlvbihlbGVtLCBwYXJlbnQsIGV2ZW50LCB3aWR0aCl7XG5cdFx0XHR2YXIgc291cmNlcywgaSwgbGVuO1xuXHRcdFx0ZWxlbS5fbGF6eXNpemVzV2lkdGggPSB3aWR0aDtcblx0XHRcdHdpZHRoICs9ICdweCc7XG5cblx0XHRcdGVsZW0uc2V0QXR0cmlidXRlKCdzaXplcycsIHdpZHRoKTtcblxuXHRcdFx0aWYocmVnUGljdHVyZS50ZXN0KHBhcmVudC5ub2RlTmFtZSB8fCAnJykpe1xuXHRcdFx0XHRzb3VyY2VzID0gcGFyZW50LmdldEVsZW1lbnRzQnlUYWdOYW1lKCdzb3VyY2UnKTtcblx0XHRcdFx0Zm9yKGkgPSAwLCBsZW4gPSBzb3VyY2VzLmxlbmd0aDsgaSA8IGxlbjsgaSsrKXtcblx0XHRcdFx0XHRzb3VyY2VzW2ldLnNldEF0dHJpYnV0ZSgnc2l6ZXMnLCB3aWR0aCk7XG5cdFx0XHRcdH1cblx0XHRcdH1cblxuXHRcdFx0aWYoIWV2ZW50LmRldGFpbC5kYXRhQXR0cil7XG5cdFx0XHRcdHVwZGF0ZVBvbHlmaWxsKGVsZW0sIGV2ZW50LmRldGFpbCk7XG5cdFx0XHR9XG5cdFx0fSk7XG5cdFx0LyoqXG5cdFx0ICpcblx0XHQgKiBAcGFyYW0gZWxlbSB7RWxlbWVudH1cblx0XHQgKiBAcGFyYW0gZGF0YUF0dHJcblx0XHQgKiBAcGFyYW0gW3dpZHRoXSB7IG51bWJlciB9XG5cdFx0ICovXG5cdFx0dmFyIGdldFNpemVFbGVtZW50ID0gZnVuY3Rpb24gKGVsZW0sIGRhdGFBdHRyLCB3aWR0aCl7XG5cdFx0XHR2YXIgZXZlbnQ7XG5cdFx0XHR2YXIgcGFyZW50ID0gZWxlbS5wYXJlbnROb2RlO1xuXG5cdFx0XHRpZihwYXJlbnQpe1xuXHRcdFx0XHR3aWR0aCA9IGdldFdpZHRoKGVsZW0sIHBhcmVudCwgd2lkdGgpO1xuXHRcdFx0XHRldmVudCA9IHRyaWdnZXJFdmVudChlbGVtLCAnbGF6eWJlZm9yZXNpemVzJywge3dpZHRoOiB3aWR0aCwgZGF0YUF0dHI6ICEhZGF0YUF0dHJ9KTtcblxuXHRcdFx0XHRpZighZXZlbnQuZGVmYXVsdFByZXZlbnRlZCl7XG5cdFx0XHRcdFx0d2lkdGggPSBldmVudC5kZXRhaWwud2lkdGg7XG5cblx0XHRcdFx0XHRpZih3aWR0aCAmJiB3aWR0aCAhPT0gZWxlbS5fbGF6eXNpemVzV2lkdGgpe1xuXHRcdFx0XHRcdFx0c2l6ZUVsZW1lbnQoZWxlbSwgcGFyZW50LCBldmVudCwgd2lkdGgpO1xuXHRcdFx0XHRcdH1cblx0XHRcdFx0fVxuXHRcdFx0fVxuXHRcdH07XG5cblx0XHR2YXIgdXBkYXRlRWxlbWVudHNTaXplcyA9IGZ1bmN0aW9uKCl7XG5cdFx0XHR2YXIgaTtcblx0XHRcdHZhciBsZW4gPSBhdXRvc2l6ZXNFbGVtcy5sZW5ndGg7XG5cdFx0XHRpZihsZW4pe1xuXHRcdFx0XHRpID0gMDtcblxuXHRcdFx0XHRmb3IoOyBpIDwgbGVuOyBpKyspe1xuXHRcdFx0XHRcdGdldFNpemVFbGVtZW50KGF1dG9zaXplc0VsZW1zW2ldKTtcblx0XHRcdFx0fVxuXHRcdFx0fVxuXHRcdH07XG5cblx0XHR2YXIgZGVib3VuY2VkVXBkYXRlRWxlbWVudHNTaXplcyA9IGRlYm91bmNlKHVwZGF0ZUVsZW1lbnRzU2l6ZXMpO1xuXG5cdFx0cmV0dXJuIHtcblx0XHRcdF86IGZ1bmN0aW9uKCl7XG5cdFx0XHRcdGF1dG9zaXplc0VsZW1zID0gZG9jdW1lbnQuZ2V0RWxlbWVudHNCeUNsYXNzTmFtZShsYXp5U2l6ZXNDZmcuYXV0b3NpemVzQ2xhc3MpO1xuXHRcdFx0XHRhZGRFdmVudExpc3RlbmVyKCdyZXNpemUnLCBkZWJvdW5jZWRVcGRhdGVFbGVtZW50c1NpemVzKTtcblx0XHRcdH0sXG5cdFx0XHRjaGVja0VsZW1zOiBkZWJvdW5jZWRVcGRhdGVFbGVtZW50c1NpemVzLFxuXHRcdFx0dXBkYXRlRWxlbTogZ2V0U2l6ZUVsZW1lbnRcblx0XHR9O1xuXHR9KSgpO1xuXG5cdHZhciBpbml0ID0gZnVuY3Rpb24oKXtcblx0XHRpZighaW5pdC5pICYmIGRvY3VtZW50LmdldEVsZW1lbnRzQnlDbGFzc05hbWUpe1xuXHRcdFx0aW5pdC5pID0gdHJ1ZTtcblx0XHRcdGF1dG9TaXplci5fKCk7XG5cdFx0XHRsb2FkZXIuXygpO1xuXHRcdH1cblx0fTtcblxuXHRzZXRUaW1lb3V0KGZ1bmN0aW9uKCl7XG5cdFx0aWYobGF6eVNpemVzQ2ZnLmluaXQpe1xuXHRcdFx0aW5pdCgpO1xuXHRcdH1cblx0fSk7XG5cblx0bGF6eXNpemVzID0ge1xuXHRcdC8qKlxuXHRcdCAqIEB0eXBlIHsgTGF6eVNpemVzQ29uZmlnUGFydGlhbCB9XG5cdFx0ICovXG5cdFx0Y2ZnOiBsYXp5U2l6ZXNDZmcsXG5cdFx0YXV0b1NpemVyOiBhdXRvU2l6ZXIsXG5cdFx0bG9hZGVyOiBsb2FkZXIsXG5cdFx0aW5pdDogaW5pdCxcblx0XHR1UDogdXBkYXRlUG9seWZpbGwsXG5cdFx0YUM6IGFkZENsYXNzLFxuXHRcdHJDOiByZW1vdmVDbGFzcyxcblx0XHRoQzogaGFzQ2xhc3MsXG5cdFx0ZmlyZTogdHJpZ2dlckV2ZW50LFxuXHRcdGdXOiBnZXRXaWR0aCxcblx0XHRyQUY6IHJBRixcblx0fTtcblxuXHRyZXR1cm4gbGF6eXNpemVzO1xufVxuKSk7XG4iLCAiKGZ1bmN0aW9uKHdpbmRvdywgZmFjdG9yeSkge1xuXHR2YXIgZ2xvYmFsSW5zdGFsbCA9IGZ1bmN0aW9uKCl7XG5cdFx0ZmFjdG9yeSh3aW5kb3cubGF6eVNpemVzKTtcblx0XHR3aW5kb3cucmVtb3ZlRXZlbnRMaXN0ZW5lcignbGF6eXVudmVpbHJlYWQnLCBnbG9iYWxJbnN0YWxsLCB0cnVlKTtcblx0fTtcblxuXHRmYWN0b3J5ID0gZmFjdG9yeS5iaW5kKG51bGwsIHdpbmRvdywgd2luZG93LmRvY3VtZW50KTtcblxuXHRpZih0eXBlb2YgbW9kdWxlID09ICdvYmplY3QnICYmIG1vZHVsZS5leHBvcnRzKXtcblx0XHRmYWN0b3J5KHJlcXVpcmUoJ2xhenlzaXplcycpKTtcblx0fSBlbHNlIGlmICh0eXBlb2YgZGVmaW5lID09ICdmdW5jdGlvbicgJiYgZGVmaW5lLmFtZCkge1xuXHRcdGRlZmluZShbJ2xhenlzaXplcyddLCBmYWN0b3J5KTtcblx0fSBlbHNlIGlmKHdpbmRvdy5sYXp5U2l6ZXMpIHtcblx0XHRnbG9iYWxJbnN0YWxsKCk7XG5cdH0gZWxzZSB7XG5cdFx0d2luZG93LmFkZEV2ZW50TGlzdGVuZXIoJ2xhenl1bnZlaWxyZWFkJywgZ2xvYmFsSW5zdGFsbCwgdHJ1ZSk7XG5cdH1cbn0od2luZG93LCBmdW5jdGlvbih3aW5kb3csIGRvY3VtZW50LCBsYXp5U2l6ZXMpIHtcblx0J3VzZSBzdHJpY3QnO1xuXG5cdHZhciBpbWdTdXBwb3J0ID0gJ2xvYWRpbmcnIGluIEhUTUxJbWFnZUVsZW1lbnQucHJvdG90eXBlO1xuXHR2YXIgaWZyYW1lU3VwcG9ydCA9ICdsb2FkaW5nJyBpbiBIVE1MSUZyYW1lRWxlbWVudC5wcm90b3R5cGU7XG5cdHZhciBpc0NvbmZpZ1NldCA9IGZhbHNlO1xuXHR2YXIgb2xkUHJlbWF0dXJlVW52ZWlsID0gbGF6eVNpemVzLnByZW1hdHVyZVVudmVpbDtcblx0dmFyIGNmZyA9IGxhenlTaXplcy5jZmc7XG5cdHZhciBsaXN0ZW5lck1hcCA9IHtcblx0XHRmb2N1czogMSxcblx0XHRtb3VzZW92ZXI6IDEsXG5cdFx0Y2xpY2s6IDEsXG5cdFx0bG9hZDogMSxcblx0XHR0cmFuc2l0aW9uZW5kOiAxLFxuXHRcdGFuaW1hdGlvbmVuZDogMSxcblx0XHRzY3JvbGw6IDEsXG5cdFx0cmVzaXplOiAxLFxuXHR9O1xuXG5cdGlmICghY2ZnLm5hdGl2ZUxvYWRpbmcpIHtcblx0XHRjZmcubmF0aXZlTG9hZGluZyA9IHt9O1xuXHR9XG5cblx0aWYgKCF3aW5kb3cuYWRkRXZlbnRMaXN0ZW5lciB8fCAhd2luZG93Lk11dGF0aW9uT2JzZXJ2ZXIgfHwgKCFpbWdTdXBwb3J0ICYmICFpZnJhbWVTdXBwb3J0KSkge1xuXHRcdHJldHVybjtcblx0fVxuXG5cdGZ1bmN0aW9uIGRpc2FibGVFdmVudHMoKSB7XG5cdFx0dmFyIGxvYWRlciA9IGxhenlTaXplcy5sb2FkZXI7XG5cdFx0dmFyIHRocm90dGxlZENoZWNrRWxlbWVudHMgPSBsb2FkZXIuY2hlY2tFbGVtcztcblx0XHR2YXIgcmVtb3ZlQUxTTCA9IGZ1bmN0aW9uKCl7XG5cdFx0XHRzZXRUaW1lb3V0KGZ1bmN0aW9uKCl7XG5cdFx0XHRcdHdpbmRvdy5yZW1vdmVFdmVudExpc3RlbmVyKCdzY3JvbGwnLCBsb2FkZXIuX2FMU0wsIHRydWUpO1xuXHRcdFx0fSwgMTAwMCk7XG5cdFx0fTtcblx0XHR2YXIgY3VycmVudExpc3RlbmVyTWFwID0gdHlwZW9mIGNmZy5uYXRpdmVMb2FkaW5nLmRpc2FibGVMaXN0ZW5lcnMgPT0gJ29iamVjdCcgP1xuXHRcdFx0Y2ZnLm5hdGl2ZUxvYWRpbmcuZGlzYWJsZUxpc3RlbmVycyA6XG5cdFx0XHRsaXN0ZW5lck1hcDtcblxuXHRcdGlmIChjdXJyZW50TGlzdGVuZXJNYXAuc2Nyb2xsKSB7XG5cdFx0XHR3aW5kb3cuYWRkRXZlbnRMaXN0ZW5lcignbG9hZCcsIHJlbW92ZUFMU0wpO1xuXHRcdFx0cmVtb3ZlQUxTTCgpO1xuXG5cdFx0XHR3aW5kb3cucmVtb3ZlRXZlbnRMaXN0ZW5lcignc2Nyb2xsJywgdGhyb3R0bGVkQ2hlY2tFbGVtZW50cywgdHJ1ZSk7XG5cdFx0fVxuXG5cdFx0aWYgKGN1cnJlbnRMaXN0ZW5lck1hcC5yZXNpemUpIHtcblx0XHRcdHdpbmRvdy5yZW1vdmVFdmVudExpc3RlbmVyKCdyZXNpemUnLCB0aHJvdHRsZWRDaGVja0VsZW1lbnRzLCB0cnVlKTtcblx0XHR9XG5cblx0XHRPYmplY3Qua2V5cyhjdXJyZW50TGlzdGVuZXJNYXApLmZvckVhY2goZnVuY3Rpb24obmFtZSkge1xuXHRcdFx0aWYgKGN1cnJlbnRMaXN0ZW5lck1hcFtuYW1lXSkge1xuXHRcdFx0XHRkb2N1bWVudC5yZW1vdmVFdmVudExpc3RlbmVyKG5hbWUsIHRocm90dGxlZENoZWNrRWxlbWVudHMsIHRydWUpO1xuXHRcdFx0fVxuXHRcdH0pO1xuXHR9XG5cblx0ZnVuY3Rpb24gcnVuQ29uZmlnKCkge1xuXHRcdGlmIChpc0NvbmZpZ1NldCkge3JldHVybjt9XG5cdFx0aXNDb25maWdTZXQgPSB0cnVlO1xuXG5cdFx0aWYgKGltZ1N1cHBvcnQgJiYgaWZyYW1lU3VwcG9ydCAmJiBjZmcubmF0aXZlTG9hZGluZy5kaXNhYmxlTGlzdGVuZXJzKSB7XG5cdFx0XHRpZiAoY2ZnLm5hdGl2ZUxvYWRpbmcuZGlzYWJsZUxpc3RlbmVycyA9PT0gdHJ1ZSkge1xuXHRcdFx0XHRjZmcubmF0aXZlTG9hZGluZy5zZXRMb2FkaW5nQXR0cmlidXRlID0gdHJ1ZTtcblx0XHRcdH1cblxuXHRcdFx0ZGlzYWJsZUV2ZW50cygpO1xuXHRcdH1cblxuXHRcdGlmIChjZmcubmF0aXZlTG9hZGluZy5zZXRMb2FkaW5nQXR0cmlidXRlKSB7XG5cdFx0XHR3aW5kb3cuYWRkRXZlbnRMaXN0ZW5lcignbGF6eWJlZm9yZXVudmVpbCcsIGZ1bmN0aW9uKGUpe1xuXHRcdFx0XHR2YXIgZWxlbWVudCA9IGUudGFyZ2V0O1xuXG5cdFx0XHRcdGlmICgnbG9hZGluZycgaW4gZWxlbWVudCAmJiAhZWxlbWVudC5nZXRBdHRyaWJ1dGUoJ2xvYWRpbmcnKSkge1xuXHRcdFx0XHRcdGVsZW1lbnQuc2V0QXR0cmlidXRlKCdsb2FkaW5nJywgJ2xhenknKTtcblx0XHRcdFx0fVxuXHRcdFx0fSwgdHJ1ZSk7XG5cdFx0fVxuXHR9XG5cblx0bGF6eVNpemVzLnByZW1hdHVyZVVudmVpbCA9IGZ1bmN0aW9uIHByZW1hdHVyZVVudmVpbChlbGVtZW50KSB7XG5cblx0XHRpZiAoIWlzQ29uZmlnU2V0KSB7XG5cdFx0XHRydW5Db25maWcoKTtcblx0XHR9XG5cblx0XHRpZiAoJ2xvYWRpbmcnIGluIGVsZW1lbnQgJiZcblx0XHRcdChjZmcubmF0aXZlTG9hZGluZy5zZXRMb2FkaW5nQXR0cmlidXRlIHx8IGVsZW1lbnQuZ2V0QXR0cmlidXRlKCdsb2FkaW5nJykpICYmXG5cdFx0XHQoZWxlbWVudC5nZXRBdHRyaWJ1dGUoJ2RhdGEtc2l6ZXMnKSAhPSAnYXV0bycgfHwgZWxlbWVudC5vZmZzZXRXaWR0aCkpIHtcblx0XHRcdHJldHVybiB0cnVlO1xuXHRcdH1cblxuXHRcdGlmIChvbGRQcmVtYXR1cmVVbnZlaWwpIHtcblx0XHRcdHJldHVybiBvbGRQcmVtYXR1cmVVbnZlaWwoZWxlbWVudCk7XG5cdFx0fVxuXHR9O1xuXG59KSk7XG4iLCAiLyohXG4gKiBjbGlwYm9hcmQuanMgdjIuMC4xMVxuICogaHR0cHM6Ly9jbGlwYm9hcmRqcy5jb20vXG4gKlxuICogTGljZW5zZWQgTUlUIFx1MDBBOSBaZW5vIFJvY2hhXG4gKi9cbihmdW5jdGlvbiB3ZWJwYWNrVW5pdmVyc2FsTW9kdWxlRGVmaW5pdGlvbihyb290LCBmYWN0b3J5KSB7XG5cdGlmKHR5cGVvZiBleHBvcnRzID09PSAnb2JqZWN0JyAmJiB0eXBlb2YgbW9kdWxlID09PSAnb2JqZWN0Jylcblx0XHRtb2R1bGUuZXhwb3J0cyA9IGZhY3RvcnkoKTtcblx0ZWxzZSBpZih0eXBlb2YgZGVmaW5lID09PSAnZnVuY3Rpb24nICYmIGRlZmluZS5hbWQpXG5cdFx0ZGVmaW5lKFtdLCBmYWN0b3J5KTtcblx0ZWxzZSBpZih0eXBlb2YgZXhwb3J0cyA9PT0gJ29iamVjdCcpXG5cdFx0ZXhwb3J0c1tcIkNsaXBib2FyZEpTXCJdID0gZmFjdG9yeSgpO1xuXHRlbHNlXG5cdFx0cm9vdFtcIkNsaXBib2FyZEpTXCJdID0gZmFjdG9yeSgpO1xufSkodGhpcywgZnVuY3Rpb24oKSB7XG5yZXR1cm4gLyoqKioqKi8gKGZ1bmN0aW9uKCkgeyAvLyB3ZWJwYWNrQm9vdHN0cmFwXG4vKioqKioqLyBcdHZhciBfX3dlYnBhY2tfbW9kdWxlc19fID0gKHtcblxuLyoqKi8gNjg2OlxuLyoqKi8gKGZ1bmN0aW9uKF9fdW51c2VkX3dlYnBhY2tfbW9kdWxlLCBfX3dlYnBhY2tfZXhwb3J0c19fLCBfX3dlYnBhY2tfcmVxdWlyZV9fKSB7XG5cblwidXNlIHN0cmljdFwiO1xuXG4vLyBFWFBPUlRTXG5fX3dlYnBhY2tfcmVxdWlyZV9fLmQoX193ZWJwYWNrX2V4cG9ydHNfXywge1xuICBcImRlZmF1bHRcIjogZnVuY3Rpb24oKSB7IHJldHVybiAvKiBiaW5kaW5nICovIGNsaXBib2FyZDsgfVxufSk7XG5cbi8vIEVYVEVSTkFMIE1PRFVMRTogLi9ub2RlX21vZHVsZXMvdGlueS1lbWl0dGVyL2luZGV4LmpzXG52YXIgdGlueV9lbWl0dGVyID0gX193ZWJwYWNrX3JlcXVpcmVfXygyNzkpO1xudmFyIHRpbnlfZW1pdHRlcl9kZWZhdWx0ID0gLyojX19QVVJFX18qL19fd2VicGFja19yZXF1aXJlX18ubih0aW55X2VtaXR0ZXIpO1xuLy8gRVhURVJOQUwgTU9EVUxFOiAuL25vZGVfbW9kdWxlcy9nb29kLWxpc3RlbmVyL3NyYy9saXN0ZW4uanNcbnZhciBsaXN0ZW4gPSBfX3dlYnBhY2tfcmVxdWlyZV9fKDM3MCk7XG52YXIgbGlzdGVuX2RlZmF1bHQgPSAvKiNfX1BVUkVfXyovX193ZWJwYWNrX3JlcXVpcmVfXy5uKGxpc3Rlbik7XG4vLyBFWFRFUk5BTCBNT0RVTEU6IC4vbm9kZV9tb2R1bGVzL3NlbGVjdC9zcmMvc2VsZWN0LmpzXG52YXIgc3JjX3NlbGVjdCA9IF9fd2VicGFja19yZXF1aXJlX18oODE3KTtcbnZhciBzZWxlY3RfZGVmYXVsdCA9IC8qI19fUFVSRV9fKi9fX3dlYnBhY2tfcmVxdWlyZV9fLm4oc3JjX3NlbGVjdCk7XG47Ly8gQ09OQ0FURU5BVEVEIE1PRFVMRTogLi9zcmMvY29tbW9uL2NvbW1hbmQuanNcbi8qKlxuICogRXhlY3V0ZXMgYSBnaXZlbiBvcGVyYXRpb24gdHlwZS5cbiAqIEBwYXJhbSB7U3RyaW5nfSB0eXBlXG4gKiBAcmV0dXJuIHtCb29sZWFufVxuICovXG5mdW5jdGlvbiBjb21tYW5kKHR5cGUpIHtcbiAgdHJ5IHtcbiAgICByZXR1cm4gZG9jdW1lbnQuZXhlY0NvbW1hbmQodHlwZSk7XG4gIH0gY2F0Y2ggKGVycikge1xuICAgIHJldHVybiBmYWxzZTtcbiAgfVxufVxuOy8vIENPTkNBVEVOQVRFRCBNT0RVTEU6IC4vc3JjL2FjdGlvbnMvY3V0LmpzXG5cblxuLyoqXG4gKiBDdXQgYWN0aW9uIHdyYXBwZXIuXG4gKiBAcGFyYW0ge1N0cmluZ3xIVE1MRWxlbWVudH0gdGFyZ2V0XG4gKiBAcmV0dXJuIHtTdHJpbmd9XG4gKi9cblxudmFyIENsaXBib2FyZEFjdGlvbkN1dCA9IGZ1bmN0aW9uIENsaXBib2FyZEFjdGlvbkN1dCh0YXJnZXQpIHtcbiAgdmFyIHNlbGVjdGVkVGV4dCA9IHNlbGVjdF9kZWZhdWx0KCkodGFyZ2V0KTtcbiAgY29tbWFuZCgnY3V0Jyk7XG4gIHJldHVybiBzZWxlY3RlZFRleHQ7XG59O1xuXG4vKiBoYXJtb255IGRlZmF1bHQgZXhwb3J0ICovIHZhciBhY3Rpb25zX2N1dCA9IChDbGlwYm9hcmRBY3Rpb25DdXQpO1xuOy8vIENPTkNBVEVOQVRFRCBNT0RVTEU6IC4vc3JjL2NvbW1vbi9jcmVhdGUtZmFrZS1lbGVtZW50LmpzXG4vKipcbiAqIENyZWF0ZXMgYSBmYWtlIHRleHRhcmVhIGVsZW1lbnQgd2l0aCBhIHZhbHVlLlxuICogQHBhcmFtIHtTdHJpbmd9IHZhbHVlXG4gKiBAcmV0dXJuIHtIVE1MRWxlbWVudH1cbiAqL1xuZnVuY3Rpb24gY3JlYXRlRmFrZUVsZW1lbnQodmFsdWUpIHtcbiAgdmFyIGlzUlRMID0gZG9jdW1lbnQuZG9jdW1lbnRFbGVtZW50LmdldEF0dHJpYnV0ZSgnZGlyJykgPT09ICdydGwnO1xuICB2YXIgZmFrZUVsZW1lbnQgPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KCd0ZXh0YXJlYScpOyAvLyBQcmV2ZW50IHpvb21pbmcgb24gaU9TXG5cbiAgZmFrZUVsZW1lbnQuc3R5bGUuZm9udFNpemUgPSAnMTJwdCc7IC8vIFJlc2V0IGJveCBtb2RlbFxuXG4gIGZha2VFbGVtZW50LnN0eWxlLmJvcmRlciA9ICcwJztcbiAgZmFrZUVsZW1lbnQuc3R5bGUucGFkZGluZyA9ICcwJztcbiAgZmFrZUVsZW1lbnQuc3R5bGUubWFyZ2luID0gJzAnOyAvLyBNb3ZlIGVsZW1lbnQgb3V0IG9mIHNjcmVlbiBob3Jpem9udGFsbHlcblxuICBmYWtlRWxlbWVudC5zdHlsZS5wb3NpdGlvbiA9ICdhYnNvbHV0ZSc7XG4gIGZha2VFbGVtZW50LnN0eWxlW2lzUlRMID8gJ3JpZ2h0JyA6ICdsZWZ0J10gPSAnLTk5OTlweCc7IC8vIE1vdmUgZWxlbWVudCB0byB0aGUgc2FtZSBwb3NpdGlvbiB2ZXJ0aWNhbGx5XG5cbiAgdmFyIHlQb3NpdGlvbiA9IHdpbmRvdy5wYWdlWU9mZnNldCB8fCBkb2N1bWVudC5kb2N1bWVudEVsZW1lbnQuc2Nyb2xsVG9wO1xuICBmYWtlRWxlbWVudC5zdHlsZS50b3AgPSBcIlwiLmNvbmNhdCh5UG9zaXRpb24sIFwicHhcIik7XG4gIGZha2VFbGVtZW50LnNldEF0dHJpYnV0ZSgncmVhZG9ubHknLCAnJyk7XG4gIGZha2VFbGVtZW50LnZhbHVlID0gdmFsdWU7XG4gIHJldHVybiBmYWtlRWxlbWVudDtcbn1cbjsvLyBDT05DQVRFTkFURUQgTU9EVUxFOiAuL3NyYy9hY3Rpb25zL2NvcHkuanNcblxuXG5cbi8qKlxuICogQ3JlYXRlIGZha2UgY29weSBhY3Rpb24gd3JhcHBlciB1c2luZyBhIGZha2UgZWxlbWVudC5cbiAqIEBwYXJhbSB7U3RyaW5nfSB0YXJnZXRcbiAqIEBwYXJhbSB7T2JqZWN0fSBvcHRpb25zXG4gKiBAcmV0dXJuIHtTdHJpbmd9XG4gKi9cblxudmFyIGZha2VDb3B5QWN0aW9uID0gZnVuY3Rpb24gZmFrZUNvcHlBY3Rpb24odmFsdWUsIG9wdGlvbnMpIHtcbiAgdmFyIGZha2VFbGVtZW50ID0gY3JlYXRlRmFrZUVsZW1lbnQodmFsdWUpO1xuICBvcHRpb25zLmNvbnRhaW5lci5hcHBlbmRDaGlsZChmYWtlRWxlbWVudCk7XG4gIHZhciBzZWxlY3RlZFRleHQgPSBzZWxlY3RfZGVmYXVsdCgpKGZha2VFbGVtZW50KTtcbiAgY29tbWFuZCgnY29weScpO1xuICBmYWtlRWxlbWVudC5yZW1vdmUoKTtcbiAgcmV0dXJuIHNlbGVjdGVkVGV4dDtcbn07XG4vKipcbiAqIENvcHkgYWN0aW9uIHdyYXBwZXIuXG4gKiBAcGFyYW0ge1N0cmluZ3xIVE1MRWxlbWVudH0gdGFyZ2V0XG4gKiBAcGFyYW0ge09iamVjdH0gb3B0aW9uc1xuICogQHJldHVybiB7U3RyaW5nfVxuICovXG5cblxudmFyIENsaXBib2FyZEFjdGlvbkNvcHkgPSBmdW5jdGlvbiBDbGlwYm9hcmRBY3Rpb25Db3B5KHRhcmdldCkge1xuICB2YXIgb3B0aW9ucyA9IGFyZ3VtZW50cy5sZW5ndGggPiAxICYmIGFyZ3VtZW50c1sxXSAhPT0gdW5kZWZpbmVkID8gYXJndW1lbnRzWzFdIDoge1xuICAgIGNvbnRhaW5lcjogZG9jdW1lbnQuYm9keVxuICB9O1xuICB2YXIgc2VsZWN0ZWRUZXh0ID0gJyc7XG5cbiAgaWYgKHR5cGVvZiB0YXJnZXQgPT09ICdzdHJpbmcnKSB7XG4gICAgc2VsZWN0ZWRUZXh0ID0gZmFrZUNvcHlBY3Rpb24odGFyZ2V0LCBvcHRpb25zKTtcbiAgfSBlbHNlIGlmICh0YXJnZXQgaW5zdGFuY2VvZiBIVE1MSW5wdXRFbGVtZW50ICYmICFbJ3RleHQnLCAnc2VhcmNoJywgJ3VybCcsICd0ZWwnLCAncGFzc3dvcmQnXS5pbmNsdWRlcyh0YXJnZXQgPT09IG51bGwgfHwgdGFyZ2V0ID09PSB2b2lkIDAgPyB2b2lkIDAgOiB0YXJnZXQudHlwZSkpIHtcbiAgICAvLyBJZiBpbnB1dCB0eXBlIGRvZXNuJ3Qgc3VwcG9ydCBgc2V0U2VsZWN0aW9uUmFuZ2VgLiBTaW11bGF0ZSBpdC4gaHR0cHM6Ly9kZXZlbG9wZXIubW96aWxsYS5vcmcvZW4tVVMvZG9jcy9XZWIvQVBJL0hUTUxJbnB1dEVsZW1lbnQvc2V0U2VsZWN0aW9uUmFuZ2VcbiAgICBzZWxlY3RlZFRleHQgPSBmYWtlQ29weUFjdGlvbih0YXJnZXQudmFsdWUsIG9wdGlvbnMpO1xuICB9IGVsc2Uge1xuICAgIHNlbGVjdGVkVGV4dCA9IHNlbGVjdF9kZWZhdWx0KCkodGFyZ2V0KTtcbiAgICBjb21tYW5kKCdjb3B5Jyk7XG4gIH1cblxuICByZXR1cm4gc2VsZWN0ZWRUZXh0O1xufTtcblxuLyogaGFybW9ueSBkZWZhdWx0IGV4cG9ydCAqLyB2YXIgYWN0aW9uc19jb3B5ID0gKENsaXBib2FyZEFjdGlvbkNvcHkpO1xuOy8vIENPTkNBVEVOQVRFRCBNT0RVTEU6IC4vc3JjL2FjdGlvbnMvZGVmYXVsdC5qc1xuZnVuY3Rpb24gX3R5cGVvZihvYmopIHsgXCJAYmFiZWwvaGVscGVycyAtIHR5cGVvZlwiOyBpZiAodHlwZW9mIFN5bWJvbCA9PT0gXCJmdW5jdGlvblwiICYmIHR5cGVvZiBTeW1ib2wuaXRlcmF0b3IgPT09IFwic3ltYm9sXCIpIHsgX3R5cGVvZiA9IGZ1bmN0aW9uIF90eXBlb2Yob2JqKSB7IHJldHVybiB0eXBlb2Ygb2JqOyB9OyB9IGVsc2UgeyBfdHlwZW9mID0gZnVuY3Rpb24gX3R5cGVvZihvYmopIHsgcmV0dXJuIG9iaiAmJiB0eXBlb2YgU3ltYm9sID09PSBcImZ1bmN0aW9uXCIgJiYgb2JqLmNvbnN0cnVjdG9yID09PSBTeW1ib2wgJiYgb2JqICE9PSBTeW1ib2wucHJvdG90eXBlID8gXCJzeW1ib2xcIiA6IHR5cGVvZiBvYmo7IH07IH0gcmV0dXJuIF90eXBlb2Yob2JqKTsgfVxuXG5cblxuLyoqXG4gKiBJbm5lciBmdW5jdGlvbiB3aGljaCBwZXJmb3JtcyBzZWxlY3Rpb24gZnJvbSBlaXRoZXIgYHRleHRgIG9yIGB0YXJnZXRgXG4gKiBwcm9wZXJ0aWVzIGFuZCB0aGVuIGV4ZWN1dGVzIGNvcHkgb3IgY3V0IG9wZXJhdGlvbnMuXG4gKiBAcGFyYW0ge09iamVjdH0gb3B0aW9uc1xuICovXG5cbnZhciBDbGlwYm9hcmRBY3Rpb25EZWZhdWx0ID0gZnVuY3Rpb24gQ2xpcGJvYXJkQWN0aW9uRGVmYXVsdCgpIHtcbiAgdmFyIG9wdGlvbnMgPSBhcmd1bWVudHMubGVuZ3RoID4gMCAmJiBhcmd1bWVudHNbMF0gIT09IHVuZGVmaW5lZCA/IGFyZ3VtZW50c1swXSA6IHt9O1xuICAvLyBEZWZpbmVzIGJhc2UgcHJvcGVydGllcyBwYXNzZWQgZnJvbSBjb25zdHJ1Y3Rvci5cbiAgdmFyIF9vcHRpb25zJGFjdGlvbiA9IG9wdGlvbnMuYWN0aW9uLFxuICAgICAgYWN0aW9uID0gX29wdGlvbnMkYWN0aW9uID09PSB2b2lkIDAgPyAnY29weScgOiBfb3B0aW9ucyRhY3Rpb24sXG4gICAgICBjb250YWluZXIgPSBvcHRpb25zLmNvbnRhaW5lcixcbiAgICAgIHRhcmdldCA9IG9wdGlvbnMudGFyZ2V0LFxuICAgICAgdGV4dCA9IG9wdGlvbnMudGV4dDsgLy8gU2V0cyB0aGUgYGFjdGlvbmAgdG8gYmUgcGVyZm9ybWVkIHdoaWNoIGNhbiBiZSBlaXRoZXIgJ2NvcHknIG9yICdjdXQnLlxuXG4gIGlmIChhY3Rpb24gIT09ICdjb3B5JyAmJiBhY3Rpb24gIT09ICdjdXQnKSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKCdJbnZhbGlkIFwiYWN0aW9uXCIgdmFsdWUsIHVzZSBlaXRoZXIgXCJjb3B5XCIgb3IgXCJjdXRcIicpO1xuICB9IC8vIFNldHMgdGhlIGB0YXJnZXRgIHByb3BlcnR5IHVzaW5nIGFuIGVsZW1lbnQgdGhhdCB3aWxsIGJlIGhhdmUgaXRzIGNvbnRlbnQgY29waWVkLlxuXG5cbiAgaWYgKHRhcmdldCAhPT0gdW5kZWZpbmVkKSB7XG4gICAgaWYgKHRhcmdldCAmJiBfdHlwZW9mKHRhcmdldCkgPT09ICdvYmplY3QnICYmIHRhcmdldC5ub2RlVHlwZSA9PT0gMSkge1xuICAgICAgaWYgKGFjdGlvbiA9PT0gJ2NvcHknICYmIHRhcmdldC5oYXNBdHRyaWJ1dGUoJ2Rpc2FibGVkJykpIHtcbiAgICAgICAgdGhyb3cgbmV3IEVycm9yKCdJbnZhbGlkIFwidGFyZ2V0XCIgYXR0cmlidXRlLiBQbGVhc2UgdXNlIFwicmVhZG9ubHlcIiBpbnN0ZWFkIG9mIFwiZGlzYWJsZWRcIiBhdHRyaWJ1dGUnKTtcbiAgICAgIH1cblxuICAgICAgaWYgKGFjdGlvbiA9PT0gJ2N1dCcgJiYgKHRhcmdldC5oYXNBdHRyaWJ1dGUoJ3JlYWRvbmx5JykgfHwgdGFyZ2V0Lmhhc0F0dHJpYnV0ZSgnZGlzYWJsZWQnKSkpIHtcbiAgICAgICAgdGhyb3cgbmV3IEVycm9yKCdJbnZhbGlkIFwidGFyZ2V0XCIgYXR0cmlidXRlLiBZb3UgY2FuXFwndCBjdXQgdGV4dCBmcm9tIGVsZW1lbnRzIHdpdGggXCJyZWFkb25seVwiIG9yIFwiZGlzYWJsZWRcIiBhdHRyaWJ1dGVzJyk7XG4gICAgICB9XG4gICAgfSBlbHNlIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcignSW52YWxpZCBcInRhcmdldFwiIHZhbHVlLCB1c2UgYSB2YWxpZCBFbGVtZW50Jyk7XG4gICAgfVxuICB9IC8vIERlZmluZSBzZWxlY3Rpb24gc3RyYXRlZ3kgYmFzZWQgb24gYHRleHRgIHByb3BlcnR5LlxuXG5cbiAgaWYgKHRleHQpIHtcbiAgICByZXR1cm4gYWN0aW9uc19jb3B5KHRleHQsIHtcbiAgICAgIGNvbnRhaW5lcjogY29udGFpbmVyXG4gICAgfSk7XG4gIH0gLy8gRGVmaW5lcyB3aGljaCBzZWxlY3Rpb24gc3RyYXRlZ3kgYmFzZWQgb24gYHRhcmdldGAgcHJvcGVydHkuXG5cblxuICBpZiAodGFyZ2V0KSB7XG4gICAgcmV0dXJuIGFjdGlvbiA9PT0gJ2N1dCcgPyBhY3Rpb25zX2N1dCh0YXJnZXQpIDogYWN0aW9uc19jb3B5KHRhcmdldCwge1xuICAgICAgY29udGFpbmVyOiBjb250YWluZXJcbiAgICB9KTtcbiAgfVxufTtcblxuLyogaGFybW9ueSBkZWZhdWx0IGV4cG9ydCAqLyB2YXIgYWN0aW9uc19kZWZhdWx0ID0gKENsaXBib2FyZEFjdGlvbkRlZmF1bHQpO1xuOy8vIENPTkNBVEVOQVRFRCBNT0RVTEU6IC4vc3JjL2NsaXBib2FyZC5qc1xuZnVuY3Rpb24gY2xpcGJvYXJkX3R5cGVvZihvYmopIHsgXCJAYmFiZWwvaGVscGVycyAtIHR5cGVvZlwiOyBpZiAodHlwZW9mIFN5bWJvbCA9PT0gXCJmdW5jdGlvblwiICYmIHR5cGVvZiBTeW1ib2wuaXRlcmF0b3IgPT09IFwic3ltYm9sXCIpIHsgY2xpcGJvYXJkX3R5cGVvZiA9IGZ1bmN0aW9uIF90eXBlb2Yob2JqKSB7IHJldHVybiB0eXBlb2Ygb2JqOyB9OyB9IGVsc2UgeyBjbGlwYm9hcmRfdHlwZW9mID0gZnVuY3Rpb24gX3R5cGVvZihvYmopIHsgcmV0dXJuIG9iaiAmJiB0eXBlb2YgU3ltYm9sID09PSBcImZ1bmN0aW9uXCIgJiYgb2JqLmNvbnN0cnVjdG9yID09PSBTeW1ib2wgJiYgb2JqICE9PSBTeW1ib2wucHJvdG90eXBlID8gXCJzeW1ib2xcIiA6IHR5cGVvZiBvYmo7IH07IH0gcmV0dXJuIGNsaXBib2FyZF90eXBlb2Yob2JqKTsgfVxuXG5mdW5jdGlvbiBfY2xhc3NDYWxsQ2hlY2soaW5zdGFuY2UsIENvbnN0cnVjdG9yKSB7IGlmICghKGluc3RhbmNlIGluc3RhbmNlb2YgQ29uc3RydWN0b3IpKSB7IHRocm93IG5ldyBUeXBlRXJyb3IoXCJDYW5ub3QgY2FsbCBhIGNsYXNzIGFzIGEgZnVuY3Rpb25cIik7IH0gfVxuXG5mdW5jdGlvbiBfZGVmaW5lUHJvcGVydGllcyh0YXJnZXQsIHByb3BzKSB7IGZvciAodmFyIGkgPSAwOyBpIDwgcHJvcHMubGVuZ3RoOyBpKyspIHsgdmFyIGRlc2NyaXB0b3IgPSBwcm9wc1tpXTsgZGVzY3JpcHRvci5lbnVtZXJhYmxlID0gZGVzY3JpcHRvci5lbnVtZXJhYmxlIHx8IGZhbHNlOyBkZXNjcmlwdG9yLmNvbmZpZ3VyYWJsZSA9IHRydWU7IGlmIChcInZhbHVlXCIgaW4gZGVzY3JpcHRvcikgZGVzY3JpcHRvci53cml0YWJsZSA9IHRydWU7IE9iamVjdC5kZWZpbmVQcm9wZXJ0eSh0YXJnZXQsIGRlc2NyaXB0b3Iua2V5LCBkZXNjcmlwdG9yKTsgfSB9XG5cbmZ1bmN0aW9uIF9jcmVhdGVDbGFzcyhDb25zdHJ1Y3RvciwgcHJvdG9Qcm9wcywgc3RhdGljUHJvcHMpIHsgaWYgKHByb3RvUHJvcHMpIF9kZWZpbmVQcm9wZXJ0aWVzKENvbnN0cnVjdG9yLnByb3RvdHlwZSwgcHJvdG9Qcm9wcyk7IGlmIChzdGF0aWNQcm9wcykgX2RlZmluZVByb3BlcnRpZXMoQ29uc3RydWN0b3IsIHN0YXRpY1Byb3BzKTsgcmV0dXJuIENvbnN0cnVjdG9yOyB9XG5cbmZ1bmN0aW9uIF9pbmhlcml0cyhzdWJDbGFzcywgc3VwZXJDbGFzcykgeyBpZiAodHlwZW9mIHN1cGVyQ2xhc3MgIT09IFwiZnVuY3Rpb25cIiAmJiBzdXBlckNsYXNzICE9PSBudWxsKSB7IHRocm93IG5ldyBUeXBlRXJyb3IoXCJTdXBlciBleHByZXNzaW9uIG11c3QgZWl0aGVyIGJlIG51bGwgb3IgYSBmdW5jdGlvblwiKTsgfSBzdWJDbGFzcy5wcm90b3R5cGUgPSBPYmplY3QuY3JlYXRlKHN1cGVyQ2xhc3MgJiYgc3VwZXJDbGFzcy5wcm90b3R5cGUsIHsgY29uc3RydWN0b3I6IHsgdmFsdWU6IHN1YkNsYXNzLCB3cml0YWJsZTogdHJ1ZSwgY29uZmlndXJhYmxlOiB0cnVlIH0gfSk7IGlmIChzdXBlckNsYXNzKSBfc2V0UHJvdG90eXBlT2Yoc3ViQ2xhc3MsIHN1cGVyQ2xhc3MpOyB9XG5cbmZ1bmN0aW9uIF9zZXRQcm90b3R5cGVPZihvLCBwKSB7IF9zZXRQcm90b3R5cGVPZiA9IE9iamVjdC5zZXRQcm90b3R5cGVPZiB8fCBmdW5jdGlvbiBfc2V0UHJvdG90eXBlT2YobywgcCkgeyBvLl9fcHJvdG9fXyA9IHA7IHJldHVybiBvOyB9OyByZXR1cm4gX3NldFByb3RvdHlwZU9mKG8sIHApOyB9XG5cbmZ1bmN0aW9uIF9jcmVhdGVTdXBlcihEZXJpdmVkKSB7IHZhciBoYXNOYXRpdmVSZWZsZWN0Q29uc3RydWN0ID0gX2lzTmF0aXZlUmVmbGVjdENvbnN0cnVjdCgpOyByZXR1cm4gZnVuY3Rpb24gX2NyZWF0ZVN1cGVySW50ZXJuYWwoKSB7IHZhciBTdXBlciA9IF9nZXRQcm90b3R5cGVPZihEZXJpdmVkKSwgcmVzdWx0OyBpZiAoaGFzTmF0aXZlUmVmbGVjdENvbnN0cnVjdCkgeyB2YXIgTmV3VGFyZ2V0ID0gX2dldFByb3RvdHlwZU9mKHRoaXMpLmNvbnN0cnVjdG9yOyByZXN1bHQgPSBSZWZsZWN0LmNvbnN0cnVjdChTdXBlciwgYXJndW1lbnRzLCBOZXdUYXJnZXQpOyB9IGVsc2UgeyByZXN1bHQgPSBTdXBlci5hcHBseSh0aGlzLCBhcmd1bWVudHMpOyB9IHJldHVybiBfcG9zc2libGVDb25zdHJ1Y3RvclJldHVybih0aGlzLCByZXN1bHQpOyB9OyB9XG5cbmZ1bmN0aW9uIF9wb3NzaWJsZUNvbnN0cnVjdG9yUmV0dXJuKHNlbGYsIGNhbGwpIHsgaWYgKGNhbGwgJiYgKGNsaXBib2FyZF90eXBlb2YoY2FsbCkgPT09IFwib2JqZWN0XCIgfHwgdHlwZW9mIGNhbGwgPT09IFwiZnVuY3Rpb25cIikpIHsgcmV0dXJuIGNhbGw7IH0gcmV0dXJuIF9hc3NlcnRUaGlzSW5pdGlhbGl6ZWQoc2VsZik7IH1cblxuZnVuY3Rpb24gX2Fzc2VydFRoaXNJbml0aWFsaXplZChzZWxmKSB7IGlmIChzZWxmID09PSB2b2lkIDApIHsgdGhyb3cgbmV3IFJlZmVyZW5jZUVycm9yKFwidGhpcyBoYXNuJ3QgYmVlbiBpbml0aWFsaXNlZCAtIHN1cGVyKCkgaGFzbid0IGJlZW4gY2FsbGVkXCIpOyB9IHJldHVybiBzZWxmOyB9XG5cbmZ1bmN0aW9uIF9pc05hdGl2ZVJlZmxlY3RDb25zdHJ1Y3QoKSB7IGlmICh0eXBlb2YgUmVmbGVjdCA9PT0gXCJ1bmRlZmluZWRcIiB8fCAhUmVmbGVjdC5jb25zdHJ1Y3QpIHJldHVybiBmYWxzZTsgaWYgKFJlZmxlY3QuY29uc3RydWN0LnNoYW0pIHJldHVybiBmYWxzZTsgaWYgKHR5cGVvZiBQcm94eSA9PT0gXCJmdW5jdGlvblwiKSByZXR1cm4gdHJ1ZTsgdHJ5IHsgRGF0ZS5wcm90b3R5cGUudG9TdHJpbmcuY2FsbChSZWZsZWN0LmNvbnN0cnVjdChEYXRlLCBbXSwgZnVuY3Rpb24gKCkge30pKTsgcmV0dXJuIHRydWU7IH0gY2F0Y2ggKGUpIHsgcmV0dXJuIGZhbHNlOyB9IH1cblxuZnVuY3Rpb24gX2dldFByb3RvdHlwZU9mKG8pIHsgX2dldFByb3RvdHlwZU9mID0gT2JqZWN0LnNldFByb3RvdHlwZU9mID8gT2JqZWN0LmdldFByb3RvdHlwZU9mIDogZnVuY3Rpb24gX2dldFByb3RvdHlwZU9mKG8pIHsgcmV0dXJuIG8uX19wcm90b19fIHx8IE9iamVjdC5nZXRQcm90b3R5cGVPZihvKTsgfTsgcmV0dXJuIF9nZXRQcm90b3R5cGVPZihvKTsgfVxuXG5cblxuXG5cblxuLyoqXG4gKiBIZWxwZXIgZnVuY3Rpb24gdG8gcmV0cmlldmUgYXR0cmlidXRlIHZhbHVlLlxuICogQHBhcmFtIHtTdHJpbmd9IHN1ZmZpeFxuICogQHBhcmFtIHtFbGVtZW50fSBlbGVtZW50XG4gKi9cblxuZnVuY3Rpb24gZ2V0QXR0cmlidXRlVmFsdWUoc3VmZml4LCBlbGVtZW50KSB7XG4gIHZhciBhdHRyaWJ1dGUgPSBcImRhdGEtY2xpcGJvYXJkLVwiLmNvbmNhdChzdWZmaXgpO1xuXG4gIGlmICghZWxlbWVudC5oYXNBdHRyaWJ1dGUoYXR0cmlidXRlKSkge1xuICAgIHJldHVybjtcbiAgfVxuXG4gIHJldHVybiBlbGVtZW50LmdldEF0dHJpYnV0ZShhdHRyaWJ1dGUpO1xufVxuLyoqXG4gKiBCYXNlIGNsYXNzIHdoaWNoIHRha2VzIG9uZSBvciBtb3JlIGVsZW1lbnRzLCBhZGRzIGV2ZW50IGxpc3RlbmVycyB0byB0aGVtLFxuICogYW5kIGluc3RhbnRpYXRlcyBhIG5ldyBgQ2xpcGJvYXJkQWN0aW9uYCBvbiBlYWNoIGNsaWNrLlxuICovXG5cblxudmFyIENsaXBib2FyZCA9IC8qI19fUFVSRV9fKi9mdW5jdGlvbiAoX0VtaXR0ZXIpIHtcbiAgX2luaGVyaXRzKENsaXBib2FyZCwgX0VtaXR0ZXIpO1xuXG4gIHZhciBfc3VwZXIgPSBfY3JlYXRlU3VwZXIoQ2xpcGJvYXJkKTtcblxuICAvKipcbiAgICogQHBhcmFtIHtTdHJpbmd8SFRNTEVsZW1lbnR8SFRNTENvbGxlY3Rpb258Tm9kZUxpc3R9IHRyaWdnZXJcbiAgICogQHBhcmFtIHtPYmplY3R9IG9wdGlvbnNcbiAgICovXG4gIGZ1bmN0aW9uIENsaXBib2FyZCh0cmlnZ2VyLCBvcHRpb25zKSB7XG4gICAgdmFyIF90aGlzO1xuXG4gICAgX2NsYXNzQ2FsbENoZWNrKHRoaXMsIENsaXBib2FyZCk7XG5cbiAgICBfdGhpcyA9IF9zdXBlci5jYWxsKHRoaXMpO1xuXG4gICAgX3RoaXMucmVzb2x2ZU9wdGlvbnMob3B0aW9ucyk7XG5cbiAgICBfdGhpcy5saXN0ZW5DbGljayh0cmlnZ2VyKTtcblxuICAgIHJldHVybiBfdGhpcztcbiAgfVxuICAvKipcbiAgICogRGVmaW5lcyBpZiBhdHRyaWJ1dGVzIHdvdWxkIGJlIHJlc29sdmVkIHVzaW5nIGludGVybmFsIHNldHRlciBmdW5jdGlvbnNcbiAgICogb3IgY3VzdG9tIGZ1bmN0aW9ucyB0aGF0IHdlcmUgcGFzc2VkIGluIHRoZSBjb25zdHJ1Y3Rvci5cbiAgICogQHBhcmFtIHtPYmplY3R9IG9wdGlvbnNcbiAgICovXG5cblxuICBfY3JlYXRlQ2xhc3MoQ2xpcGJvYXJkLCBbe1xuICAgIGtleTogXCJyZXNvbHZlT3B0aW9uc1wiLFxuICAgIHZhbHVlOiBmdW5jdGlvbiByZXNvbHZlT3B0aW9ucygpIHtcbiAgICAgIHZhciBvcHRpb25zID0gYXJndW1lbnRzLmxlbmd0aCA+IDAgJiYgYXJndW1lbnRzWzBdICE9PSB1bmRlZmluZWQgPyBhcmd1bWVudHNbMF0gOiB7fTtcbiAgICAgIHRoaXMuYWN0aW9uID0gdHlwZW9mIG9wdGlvbnMuYWN0aW9uID09PSAnZnVuY3Rpb24nID8gb3B0aW9ucy5hY3Rpb24gOiB0aGlzLmRlZmF1bHRBY3Rpb247XG4gICAgICB0aGlzLnRhcmdldCA9IHR5cGVvZiBvcHRpb25zLnRhcmdldCA9PT0gJ2Z1bmN0aW9uJyA/IG9wdGlvbnMudGFyZ2V0IDogdGhpcy5kZWZhdWx0VGFyZ2V0O1xuICAgICAgdGhpcy50ZXh0ID0gdHlwZW9mIG9wdGlvbnMudGV4dCA9PT0gJ2Z1bmN0aW9uJyA/IG9wdGlvbnMudGV4dCA6IHRoaXMuZGVmYXVsdFRleHQ7XG4gICAgICB0aGlzLmNvbnRhaW5lciA9IGNsaXBib2FyZF90eXBlb2Yob3B0aW9ucy5jb250YWluZXIpID09PSAnb2JqZWN0JyA/IG9wdGlvbnMuY29udGFpbmVyIDogZG9jdW1lbnQuYm9keTtcbiAgICB9XG4gICAgLyoqXG4gICAgICogQWRkcyBhIGNsaWNrIGV2ZW50IGxpc3RlbmVyIHRvIHRoZSBwYXNzZWQgdHJpZ2dlci5cbiAgICAgKiBAcGFyYW0ge1N0cmluZ3xIVE1MRWxlbWVudHxIVE1MQ29sbGVjdGlvbnxOb2RlTGlzdH0gdHJpZ2dlclxuICAgICAqL1xuXG4gIH0sIHtcbiAgICBrZXk6IFwibGlzdGVuQ2xpY2tcIixcbiAgICB2YWx1ZTogZnVuY3Rpb24gbGlzdGVuQ2xpY2sodHJpZ2dlcikge1xuICAgICAgdmFyIF90aGlzMiA9IHRoaXM7XG5cbiAgICAgIHRoaXMubGlzdGVuZXIgPSBsaXN0ZW5fZGVmYXVsdCgpKHRyaWdnZXIsICdjbGljaycsIGZ1bmN0aW9uIChlKSB7XG4gICAgICAgIHJldHVybiBfdGhpczIub25DbGljayhlKTtcbiAgICAgIH0pO1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBEZWZpbmVzIGEgbmV3IGBDbGlwYm9hcmRBY3Rpb25gIG9uIGVhY2ggY2xpY2sgZXZlbnQuXG4gICAgICogQHBhcmFtIHtFdmVudH0gZVxuICAgICAqL1xuXG4gIH0sIHtcbiAgICBrZXk6IFwib25DbGlja1wiLFxuICAgIHZhbHVlOiBmdW5jdGlvbiBvbkNsaWNrKGUpIHtcbiAgICAgIHZhciB0cmlnZ2VyID0gZS5kZWxlZ2F0ZVRhcmdldCB8fCBlLmN1cnJlbnRUYXJnZXQ7XG4gICAgICB2YXIgYWN0aW9uID0gdGhpcy5hY3Rpb24odHJpZ2dlcikgfHwgJ2NvcHknO1xuICAgICAgdmFyIHRleHQgPSBhY3Rpb25zX2RlZmF1bHQoe1xuICAgICAgICBhY3Rpb246IGFjdGlvbixcbiAgICAgICAgY29udGFpbmVyOiB0aGlzLmNvbnRhaW5lcixcbiAgICAgICAgdGFyZ2V0OiB0aGlzLnRhcmdldCh0cmlnZ2VyKSxcbiAgICAgICAgdGV4dDogdGhpcy50ZXh0KHRyaWdnZXIpXG4gICAgICB9KTsgLy8gRmlyZXMgYW4gZXZlbnQgYmFzZWQgb24gdGhlIGNvcHkgb3BlcmF0aW9uIHJlc3VsdC5cblxuICAgICAgdGhpcy5lbWl0KHRleHQgPyAnc3VjY2VzcycgOiAnZXJyb3InLCB7XG4gICAgICAgIGFjdGlvbjogYWN0aW9uLFxuICAgICAgICB0ZXh0OiB0ZXh0LFxuICAgICAgICB0cmlnZ2VyOiB0cmlnZ2VyLFxuICAgICAgICBjbGVhclNlbGVjdGlvbjogZnVuY3Rpb24gY2xlYXJTZWxlY3Rpb24oKSB7XG4gICAgICAgICAgaWYgKHRyaWdnZXIpIHtcbiAgICAgICAgICAgIHRyaWdnZXIuZm9jdXMoKTtcbiAgICAgICAgICB9XG5cbiAgICAgICAgICB3aW5kb3cuZ2V0U2VsZWN0aW9uKCkucmVtb3ZlQWxsUmFuZ2VzKCk7XG4gICAgICAgIH1cbiAgICAgIH0pO1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBEZWZhdWx0IGBhY3Rpb25gIGxvb2t1cCBmdW5jdGlvbi5cbiAgICAgKiBAcGFyYW0ge0VsZW1lbnR9IHRyaWdnZXJcbiAgICAgKi9cblxuICB9LCB7XG4gICAga2V5OiBcImRlZmF1bHRBY3Rpb25cIixcbiAgICB2YWx1ZTogZnVuY3Rpb24gZGVmYXVsdEFjdGlvbih0cmlnZ2VyKSB7XG4gICAgICByZXR1cm4gZ2V0QXR0cmlidXRlVmFsdWUoJ2FjdGlvbicsIHRyaWdnZXIpO1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBEZWZhdWx0IGB0YXJnZXRgIGxvb2t1cCBmdW5jdGlvbi5cbiAgICAgKiBAcGFyYW0ge0VsZW1lbnR9IHRyaWdnZXJcbiAgICAgKi9cblxuICB9LCB7XG4gICAga2V5OiBcImRlZmF1bHRUYXJnZXRcIixcbiAgICB2YWx1ZTogZnVuY3Rpb24gZGVmYXVsdFRhcmdldCh0cmlnZ2VyKSB7XG4gICAgICB2YXIgc2VsZWN0b3IgPSBnZXRBdHRyaWJ1dGVWYWx1ZSgndGFyZ2V0JywgdHJpZ2dlcik7XG5cbiAgICAgIGlmIChzZWxlY3Rvcikge1xuICAgICAgICByZXR1cm4gZG9jdW1lbnQucXVlcnlTZWxlY3RvcihzZWxlY3Rvcik7XG4gICAgICB9XG4gICAgfVxuICAgIC8qKlxuICAgICAqIEFsbG93IGZpcmUgcHJvZ3JhbW1hdGljYWxseSBhIGNvcHkgYWN0aW9uXG4gICAgICogQHBhcmFtIHtTdHJpbmd8SFRNTEVsZW1lbnR9IHRhcmdldFxuICAgICAqIEBwYXJhbSB7T2JqZWN0fSBvcHRpb25zXG4gICAgICogQHJldHVybnMgVGV4dCBjb3BpZWQuXG4gICAgICovXG5cbiAgfSwge1xuICAgIGtleTogXCJkZWZhdWx0VGV4dFwiLFxuXG4gICAgLyoqXG4gICAgICogRGVmYXVsdCBgdGV4dGAgbG9va3VwIGZ1bmN0aW9uLlxuICAgICAqIEBwYXJhbSB7RWxlbWVudH0gdHJpZ2dlclxuICAgICAqL1xuICAgIHZhbHVlOiBmdW5jdGlvbiBkZWZhdWx0VGV4dCh0cmlnZ2VyKSB7XG4gICAgICByZXR1cm4gZ2V0QXR0cmlidXRlVmFsdWUoJ3RleHQnLCB0cmlnZ2VyKTtcbiAgICB9XG4gICAgLyoqXG4gICAgICogRGVzdHJveSBsaWZlY3ljbGUuXG4gICAgICovXG5cbiAgfSwge1xuICAgIGtleTogXCJkZXN0cm95XCIsXG4gICAgdmFsdWU6IGZ1bmN0aW9uIGRlc3Ryb3koKSB7XG4gICAgICB0aGlzLmxpc3RlbmVyLmRlc3Ryb3koKTtcbiAgICB9XG4gIH1dLCBbe1xuICAgIGtleTogXCJjb3B5XCIsXG4gICAgdmFsdWU6IGZ1bmN0aW9uIGNvcHkodGFyZ2V0KSB7XG4gICAgICB2YXIgb3B0aW9ucyA9IGFyZ3VtZW50cy5sZW5ndGggPiAxICYmIGFyZ3VtZW50c1sxXSAhPT0gdW5kZWZpbmVkID8gYXJndW1lbnRzWzFdIDoge1xuICAgICAgICBjb250YWluZXI6IGRvY3VtZW50LmJvZHlcbiAgICAgIH07XG4gICAgICByZXR1cm4gYWN0aW9uc19jb3B5KHRhcmdldCwgb3B0aW9ucyk7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIEFsbG93IGZpcmUgcHJvZ3JhbW1hdGljYWxseSBhIGN1dCBhY3Rpb25cbiAgICAgKiBAcGFyYW0ge1N0cmluZ3xIVE1MRWxlbWVudH0gdGFyZ2V0XG4gICAgICogQHJldHVybnMgVGV4dCBjdXR0ZWQuXG4gICAgICovXG5cbiAgfSwge1xuICAgIGtleTogXCJjdXRcIixcbiAgICB2YWx1ZTogZnVuY3Rpb24gY3V0KHRhcmdldCkge1xuICAgICAgcmV0dXJuIGFjdGlvbnNfY3V0KHRhcmdldCk7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIFJldHVybnMgdGhlIHN1cHBvcnQgb2YgdGhlIGdpdmVuIGFjdGlvbiwgb3IgYWxsIGFjdGlvbnMgaWYgbm8gYWN0aW9uIGlzXG4gICAgICogZ2l2ZW4uXG4gICAgICogQHBhcmFtIHtTdHJpbmd9IFthY3Rpb25dXG4gICAgICovXG5cbiAgfSwge1xuICAgIGtleTogXCJpc1N1cHBvcnRlZFwiLFxuICAgIHZhbHVlOiBmdW5jdGlvbiBpc1N1cHBvcnRlZCgpIHtcbiAgICAgIHZhciBhY3Rpb24gPSBhcmd1bWVudHMubGVuZ3RoID4gMCAmJiBhcmd1bWVudHNbMF0gIT09IHVuZGVmaW5lZCA/IGFyZ3VtZW50c1swXSA6IFsnY29weScsICdjdXQnXTtcbiAgICAgIHZhciBhY3Rpb25zID0gdHlwZW9mIGFjdGlvbiA9PT0gJ3N0cmluZycgPyBbYWN0aW9uXSA6IGFjdGlvbjtcbiAgICAgIHZhciBzdXBwb3J0ID0gISFkb2N1bWVudC5xdWVyeUNvbW1hbmRTdXBwb3J0ZWQ7XG4gICAgICBhY3Rpb25zLmZvckVhY2goZnVuY3Rpb24gKGFjdGlvbikge1xuICAgICAgICBzdXBwb3J0ID0gc3VwcG9ydCAmJiAhIWRvY3VtZW50LnF1ZXJ5Q29tbWFuZFN1cHBvcnRlZChhY3Rpb24pO1xuICAgICAgfSk7XG4gICAgICByZXR1cm4gc3VwcG9ydDtcbiAgICB9XG4gIH1dKTtcblxuICByZXR1cm4gQ2xpcGJvYXJkO1xufSgodGlueV9lbWl0dGVyX2RlZmF1bHQoKSkpO1xuXG4vKiBoYXJtb255IGRlZmF1bHQgZXhwb3J0ICovIHZhciBjbGlwYm9hcmQgPSAoQ2xpcGJvYXJkKTtcblxuLyoqKi8gfSksXG5cbi8qKiovIDgyODpcbi8qKiovIChmdW5jdGlvbihtb2R1bGUpIHtcblxudmFyIERPQ1VNRU5UX05PREVfVFlQRSA9IDk7XG5cbi8qKlxuICogQSBwb2x5ZmlsbCBmb3IgRWxlbWVudC5tYXRjaGVzKClcbiAqL1xuaWYgKHR5cGVvZiBFbGVtZW50ICE9PSAndW5kZWZpbmVkJyAmJiAhRWxlbWVudC5wcm90b3R5cGUubWF0Y2hlcykge1xuICAgIHZhciBwcm90byA9IEVsZW1lbnQucHJvdG90eXBlO1xuXG4gICAgcHJvdG8ubWF0Y2hlcyA9IHByb3RvLm1hdGNoZXNTZWxlY3RvciB8fFxuICAgICAgICAgICAgICAgICAgICBwcm90by5tb3pNYXRjaGVzU2VsZWN0b3IgfHxcbiAgICAgICAgICAgICAgICAgICAgcHJvdG8ubXNNYXRjaGVzU2VsZWN0b3IgfHxcbiAgICAgICAgICAgICAgICAgICAgcHJvdG8ub01hdGNoZXNTZWxlY3RvciB8fFxuICAgICAgICAgICAgICAgICAgICBwcm90by53ZWJraXRNYXRjaGVzU2VsZWN0b3I7XG59XG5cbi8qKlxuICogRmluZHMgdGhlIGNsb3Nlc3QgcGFyZW50IHRoYXQgbWF0Y2hlcyBhIHNlbGVjdG9yLlxuICpcbiAqIEBwYXJhbSB7RWxlbWVudH0gZWxlbWVudFxuICogQHBhcmFtIHtTdHJpbmd9IHNlbGVjdG9yXG4gKiBAcmV0dXJuIHtGdW5jdGlvbn1cbiAqL1xuZnVuY3Rpb24gY2xvc2VzdCAoZWxlbWVudCwgc2VsZWN0b3IpIHtcbiAgICB3aGlsZSAoZWxlbWVudCAmJiBlbGVtZW50Lm5vZGVUeXBlICE9PSBET0NVTUVOVF9OT0RFX1RZUEUpIHtcbiAgICAgICAgaWYgKHR5cGVvZiBlbGVtZW50Lm1hdGNoZXMgPT09ICdmdW5jdGlvbicgJiZcbiAgICAgICAgICAgIGVsZW1lbnQubWF0Y2hlcyhzZWxlY3RvcikpIHtcbiAgICAgICAgICByZXR1cm4gZWxlbWVudDtcbiAgICAgICAgfVxuICAgICAgICBlbGVtZW50ID0gZWxlbWVudC5wYXJlbnROb2RlO1xuICAgIH1cbn1cblxubW9kdWxlLmV4cG9ydHMgPSBjbG9zZXN0O1xuXG5cbi8qKiovIH0pLFxuXG4vKioqLyA0Mzg6XG4vKioqLyAoZnVuY3Rpb24obW9kdWxlLCBfX3VudXNlZF93ZWJwYWNrX2V4cG9ydHMsIF9fd2VicGFja19yZXF1aXJlX18pIHtcblxudmFyIGNsb3Nlc3QgPSBfX3dlYnBhY2tfcmVxdWlyZV9fKDgyOCk7XG5cbi8qKlxuICogRGVsZWdhdGVzIGV2ZW50IHRvIGEgc2VsZWN0b3IuXG4gKlxuICogQHBhcmFtIHtFbGVtZW50fSBlbGVtZW50XG4gKiBAcGFyYW0ge1N0cmluZ30gc2VsZWN0b3JcbiAqIEBwYXJhbSB7U3RyaW5nfSB0eXBlXG4gKiBAcGFyYW0ge0Z1bmN0aW9ufSBjYWxsYmFja1xuICogQHBhcmFtIHtCb29sZWFufSB1c2VDYXB0dXJlXG4gKiBAcmV0dXJuIHtPYmplY3R9XG4gKi9cbmZ1bmN0aW9uIF9kZWxlZ2F0ZShlbGVtZW50LCBzZWxlY3RvciwgdHlwZSwgY2FsbGJhY2ssIHVzZUNhcHR1cmUpIHtcbiAgICB2YXIgbGlzdGVuZXJGbiA9IGxpc3RlbmVyLmFwcGx5KHRoaXMsIGFyZ3VtZW50cyk7XG5cbiAgICBlbGVtZW50LmFkZEV2ZW50TGlzdGVuZXIodHlwZSwgbGlzdGVuZXJGbiwgdXNlQ2FwdHVyZSk7XG5cbiAgICByZXR1cm4ge1xuICAgICAgICBkZXN0cm95OiBmdW5jdGlvbigpIHtcbiAgICAgICAgICAgIGVsZW1lbnQucmVtb3ZlRXZlbnRMaXN0ZW5lcih0eXBlLCBsaXN0ZW5lckZuLCB1c2VDYXB0dXJlKTtcbiAgICAgICAgfVxuICAgIH1cbn1cblxuLyoqXG4gKiBEZWxlZ2F0ZXMgZXZlbnQgdG8gYSBzZWxlY3Rvci5cbiAqXG4gKiBAcGFyYW0ge0VsZW1lbnR8U3RyaW5nfEFycmF5fSBbZWxlbWVudHNdXG4gKiBAcGFyYW0ge1N0cmluZ30gc2VsZWN0b3JcbiAqIEBwYXJhbSB7U3RyaW5nfSB0eXBlXG4gKiBAcGFyYW0ge0Z1bmN0aW9ufSBjYWxsYmFja1xuICogQHBhcmFtIHtCb29sZWFufSB1c2VDYXB0dXJlXG4gKiBAcmV0dXJuIHtPYmplY3R9XG4gKi9cbmZ1bmN0aW9uIGRlbGVnYXRlKGVsZW1lbnRzLCBzZWxlY3RvciwgdHlwZSwgY2FsbGJhY2ssIHVzZUNhcHR1cmUpIHtcbiAgICAvLyBIYW5kbGUgdGhlIHJlZ3VsYXIgRWxlbWVudCB1c2FnZVxuICAgIGlmICh0eXBlb2YgZWxlbWVudHMuYWRkRXZlbnRMaXN0ZW5lciA9PT0gJ2Z1bmN0aW9uJykge1xuICAgICAgICByZXR1cm4gX2RlbGVnYXRlLmFwcGx5KG51bGwsIGFyZ3VtZW50cyk7XG4gICAgfVxuXG4gICAgLy8gSGFuZGxlIEVsZW1lbnQtbGVzcyB1c2FnZSwgaXQgZGVmYXVsdHMgdG8gZ2xvYmFsIGRlbGVnYXRpb25cbiAgICBpZiAodHlwZW9mIHR5cGUgPT09ICdmdW5jdGlvbicpIHtcbiAgICAgICAgLy8gVXNlIGBkb2N1bWVudGAgYXMgdGhlIGZpcnN0IHBhcmFtZXRlciwgdGhlbiBhcHBseSBhcmd1bWVudHNcbiAgICAgICAgLy8gVGhpcyBpcyBhIHNob3J0IHdheSB0byAudW5zaGlmdCBgYXJndW1lbnRzYCB3aXRob3V0IHJ1bm5pbmcgaW50byBkZW9wdGltaXphdGlvbnNcbiAgICAgICAgcmV0dXJuIF9kZWxlZ2F0ZS5iaW5kKG51bGwsIGRvY3VtZW50KS5hcHBseShudWxsLCBhcmd1bWVudHMpO1xuICAgIH1cblxuICAgIC8vIEhhbmRsZSBTZWxlY3Rvci1iYXNlZCB1c2FnZVxuICAgIGlmICh0eXBlb2YgZWxlbWVudHMgPT09ICdzdHJpbmcnKSB7XG4gICAgICAgIGVsZW1lbnRzID0gZG9jdW1lbnQucXVlcnlTZWxlY3RvckFsbChlbGVtZW50cyk7XG4gICAgfVxuXG4gICAgLy8gSGFuZGxlIEFycmF5LWxpa2UgYmFzZWQgdXNhZ2VcbiAgICByZXR1cm4gQXJyYXkucHJvdG90eXBlLm1hcC5jYWxsKGVsZW1lbnRzLCBmdW5jdGlvbiAoZWxlbWVudCkge1xuICAgICAgICByZXR1cm4gX2RlbGVnYXRlKGVsZW1lbnQsIHNlbGVjdG9yLCB0eXBlLCBjYWxsYmFjaywgdXNlQ2FwdHVyZSk7XG4gICAgfSk7XG59XG5cbi8qKlxuICogRmluZHMgY2xvc2VzdCBtYXRjaCBhbmQgaW52b2tlcyBjYWxsYmFjay5cbiAqXG4gKiBAcGFyYW0ge0VsZW1lbnR9IGVsZW1lbnRcbiAqIEBwYXJhbSB7U3RyaW5nfSBzZWxlY3RvclxuICogQHBhcmFtIHtTdHJpbmd9IHR5cGVcbiAqIEBwYXJhbSB7RnVuY3Rpb259IGNhbGxiYWNrXG4gKiBAcmV0dXJuIHtGdW5jdGlvbn1cbiAqL1xuZnVuY3Rpb24gbGlzdGVuZXIoZWxlbWVudCwgc2VsZWN0b3IsIHR5cGUsIGNhbGxiYWNrKSB7XG4gICAgcmV0dXJuIGZ1bmN0aW9uKGUpIHtcbiAgICAgICAgZS5kZWxlZ2F0ZVRhcmdldCA9IGNsb3Nlc3QoZS50YXJnZXQsIHNlbGVjdG9yKTtcblxuICAgICAgICBpZiAoZS5kZWxlZ2F0ZVRhcmdldCkge1xuICAgICAgICAgICAgY2FsbGJhY2suY2FsbChlbGVtZW50LCBlKTtcbiAgICAgICAgfVxuICAgIH1cbn1cblxubW9kdWxlLmV4cG9ydHMgPSBkZWxlZ2F0ZTtcblxuXG4vKioqLyB9KSxcblxuLyoqKi8gODc5OlxuLyoqKi8gKGZ1bmN0aW9uKF9fdW51c2VkX3dlYnBhY2tfbW9kdWxlLCBleHBvcnRzKSB7XG5cbi8qKlxuICogQ2hlY2sgaWYgYXJndW1lbnQgaXMgYSBIVE1MIGVsZW1lbnQuXG4gKlxuICogQHBhcmFtIHtPYmplY3R9IHZhbHVlXG4gKiBAcmV0dXJuIHtCb29sZWFufVxuICovXG5leHBvcnRzLm5vZGUgPSBmdW5jdGlvbih2YWx1ZSkge1xuICAgIHJldHVybiB2YWx1ZSAhPT0gdW5kZWZpbmVkXG4gICAgICAgICYmIHZhbHVlIGluc3RhbmNlb2YgSFRNTEVsZW1lbnRcbiAgICAgICAgJiYgdmFsdWUubm9kZVR5cGUgPT09IDE7XG59O1xuXG4vKipcbiAqIENoZWNrIGlmIGFyZ3VtZW50IGlzIGEgbGlzdCBvZiBIVE1MIGVsZW1lbnRzLlxuICpcbiAqIEBwYXJhbSB7T2JqZWN0fSB2YWx1ZVxuICogQHJldHVybiB7Qm9vbGVhbn1cbiAqL1xuZXhwb3J0cy5ub2RlTGlzdCA9IGZ1bmN0aW9uKHZhbHVlKSB7XG4gICAgdmFyIHR5cGUgPSBPYmplY3QucHJvdG90eXBlLnRvU3RyaW5nLmNhbGwodmFsdWUpO1xuXG4gICAgcmV0dXJuIHZhbHVlICE9PSB1bmRlZmluZWRcbiAgICAgICAgJiYgKHR5cGUgPT09ICdbb2JqZWN0IE5vZGVMaXN0XScgfHwgdHlwZSA9PT0gJ1tvYmplY3QgSFRNTENvbGxlY3Rpb25dJylcbiAgICAgICAgJiYgKCdsZW5ndGgnIGluIHZhbHVlKVxuICAgICAgICAmJiAodmFsdWUubGVuZ3RoID09PSAwIHx8IGV4cG9ydHMubm9kZSh2YWx1ZVswXSkpO1xufTtcblxuLyoqXG4gKiBDaGVjayBpZiBhcmd1bWVudCBpcyBhIHN0cmluZy5cbiAqXG4gKiBAcGFyYW0ge09iamVjdH0gdmFsdWVcbiAqIEByZXR1cm4ge0Jvb2xlYW59XG4gKi9cbmV4cG9ydHMuc3RyaW5nID0gZnVuY3Rpb24odmFsdWUpIHtcbiAgICByZXR1cm4gdHlwZW9mIHZhbHVlID09PSAnc3RyaW5nJ1xuICAgICAgICB8fCB2YWx1ZSBpbnN0YW5jZW9mIFN0cmluZztcbn07XG5cbi8qKlxuICogQ2hlY2sgaWYgYXJndW1lbnQgaXMgYSBmdW5jdGlvbi5cbiAqXG4gKiBAcGFyYW0ge09iamVjdH0gdmFsdWVcbiAqIEByZXR1cm4ge0Jvb2xlYW59XG4gKi9cbmV4cG9ydHMuZm4gPSBmdW5jdGlvbih2YWx1ZSkge1xuICAgIHZhciB0eXBlID0gT2JqZWN0LnByb3RvdHlwZS50b1N0cmluZy5jYWxsKHZhbHVlKTtcblxuICAgIHJldHVybiB0eXBlID09PSAnW29iamVjdCBGdW5jdGlvbl0nO1xufTtcblxuXG4vKioqLyB9KSxcblxuLyoqKi8gMzcwOlxuLyoqKi8gKGZ1bmN0aW9uKG1vZHVsZSwgX191bnVzZWRfd2VicGFja19leHBvcnRzLCBfX3dlYnBhY2tfcmVxdWlyZV9fKSB7XG5cbnZhciBpcyA9IF9fd2VicGFja19yZXF1aXJlX18oODc5KTtcbnZhciBkZWxlZ2F0ZSA9IF9fd2VicGFja19yZXF1aXJlX18oNDM4KTtcblxuLyoqXG4gKiBWYWxpZGF0ZXMgYWxsIHBhcmFtcyBhbmQgY2FsbHMgdGhlIHJpZ2h0XG4gKiBsaXN0ZW5lciBmdW5jdGlvbiBiYXNlZCBvbiBpdHMgdGFyZ2V0IHR5cGUuXG4gKlxuICogQHBhcmFtIHtTdHJpbmd8SFRNTEVsZW1lbnR8SFRNTENvbGxlY3Rpb258Tm9kZUxpc3R9IHRhcmdldFxuICogQHBhcmFtIHtTdHJpbmd9IHR5cGVcbiAqIEBwYXJhbSB7RnVuY3Rpb259IGNhbGxiYWNrXG4gKiBAcmV0dXJuIHtPYmplY3R9XG4gKi9cbmZ1bmN0aW9uIGxpc3Rlbih0YXJnZXQsIHR5cGUsIGNhbGxiYWNrKSB7XG4gICAgaWYgKCF0YXJnZXQgJiYgIXR5cGUgJiYgIWNhbGxiYWNrKSB7XG4gICAgICAgIHRocm93IG5ldyBFcnJvcignTWlzc2luZyByZXF1aXJlZCBhcmd1bWVudHMnKTtcbiAgICB9XG5cbiAgICBpZiAoIWlzLnN0cmluZyh0eXBlKSkge1xuICAgICAgICB0aHJvdyBuZXcgVHlwZUVycm9yKCdTZWNvbmQgYXJndW1lbnQgbXVzdCBiZSBhIFN0cmluZycpO1xuICAgIH1cblxuICAgIGlmICghaXMuZm4oY2FsbGJhY2spKSB7XG4gICAgICAgIHRocm93IG5ldyBUeXBlRXJyb3IoJ1RoaXJkIGFyZ3VtZW50IG11c3QgYmUgYSBGdW5jdGlvbicpO1xuICAgIH1cblxuICAgIGlmIChpcy5ub2RlKHRhcmdldCkpIHtcbiAgICAgICAgcmV0dXJuIGxpc3Rlbk5vZGUodGFyZ2V0LCB0eXBlLCBjYWxsYmFjayk7XG4gICAgfVxuICAgIGVsc2UgaWYgKGlzLm5vZGVMaXN0KHRhcmdldCkpIHtcbiAgICAgICAgcmV0dXJuIGxpc3Rlbk5vZGVMaXN0KHRhcmdldCwgdHlwZSwgY2FsbGJhY2spO1xuICAgIH1cbiAgICBlbHNlIGlmIChpcy5zdHJpbmcodGFyZ2V0KSkge1xuICAgICAgICByZXR1cm4gbGlzdGVuU2VsZWN0b3IodGFyZ2V0LCB0eXBlLCBjYWxsYmFjayk7XG4gICAgfVxuICAgIGVsc2Uge1xuICAgICAgICB0aHJvdyBuZXcgVHlwZUVycm9yKCdGaXJzdCBhcmd1bWVudCBtdXN0IGJlIGEgU3RyaW5nLCBIVE1MRWxlbWVudCwgSFRNTENvbGxlY3Rpb24sIG9yIE5vZGVMaXN0Jyk7XG4gICAgfVxufVxuXG4vKipcbiAqIEFkZHMgYW4gZXZlbnQgbGlzdGVuZXIgdG8gYSBIVE1MIGVsZW1lbnRcbiAqIGFuZCByZXR1cm5zIGEgcmVtb3ZlIGxpc3RlbmVyIGZ1bmN0aW9uLlxuICpcbiAqIEBwYXJhbSB7SFRNTEVsZW1lbnR9IG5vZGVcbiAqIEBwYXJhbSB7U3RyaW5nfSB0eXBlXG4gKiBAcGFyYW0ge0Z1bmN0aW9ufSBjYWxsYmFja1xuICogQHJldHVybiB7T2JqZWN0fVxuICovXG5mdW5jdGlvbiBsaXN0ZW5Ob2RlKG5vZGUsIHR5cGUsIGNhbGxiYWNrKSB7XG4gICAgbm9kZS5hZGRFdmVudExpc3RlbmVyKHR5cGUsIGNhbGxiYWNrKTtcblxuICAgIHJldHVybiB7XG4gICAgICAgIGRlc3Ryb3k6IGZ1bmN0aW9uKCkge1xuICAgICAgICAgICAgbm9kZS5yZW1vdmVFdmVudExpc3RlbmVyKHR5cGUsIGNhbGxiYWNrKTtcbiAgICAgICAgfVxuICAgIH1cbn1cblxuLyoqXG4gKiBBZGQgYW4gZXZlbnQgbGlzdGVuZXIgdG8gYSBsaXN0IG9mIEhUTUwgZWxlbWVudHNcbiAqIGFuZCByZXR1cm5zIGEgcmVtb3ZlIGxpc3RlbmVyIGZ1bmN0aW9uLlxuICpcbiAqIEBwYXJhbSB7Tm9kZUxpc3R8SFRNTENvbGxlY3Rpb259IG5vZGVMaXN0XG4gKiBAcGFyYW0ge1N0cmluZ30gdHlwZVxuICogQHBhcmFtIHtGdW5jdGlvbn0gY2FsbGJhY2tcbiAqIEByZXR1cm4ge09iamVjdH1cbiAqL1xuZnVuY3Rpb24gbGlzdGVuTm9kZUxpc3Qobm9kZUxpc3QsIHR5cGUsIGNhbGxiYWNrKSB7XG4gICAgQXJyYXkucHJvdG90eXBlLmZvckVhY2guY2FsbChub2RlTGlzdCwgZnVuY3Rpb24obm9kZSkge1xuICAgICAgICBub2RlLmFkZEV2ZW50TGlzdGVuZXIodHlwZSwgY2FsbGJhY2spO1xuICAgIH0pO1xuXG4gICAgcmV0dXJuIHtcbiAgICAgICAgZGVzdHJveTogZnVuY3Rpb24oKSB7XG4gICAgICAgICAgICBBcnJheS5wcm90b3R5cGUuZm9yRWFjaC5jYWxsKG5vZGVMaXN0LCBmdW5jdGlvbihub2RlKSB7XG4gICAgICAgICAgICAgICAgbm9kZS5yZW1vdmVFdmVudExpc3RlbmVyKHR5cGUsIGNhbGxiYWNrKTtcbiAgICAgICAgICAgIH0pO1xuICAgICAgICB9XG4gICAgfVxufVxuXG4vKipcbiAqIEFkZCBhbiBldmVudCBsaXN0ZW5lciB0byBhIHNlbGVjdG9yXG4gKiBhbmQgcmV0dXJucyBhIHJlbW92ZSBsaXN0ZW5lciBmdW5jdGlvbi5cbiAqXG4gKiBAcGFyYW0ge1N0cmluZ30gc2VsZWN0b3JcbiAqIEBwYXJhbSB7U3RyaW5nfSB0eXBlXG4gKiBAcGFyYW0ge0Z1bmN0aW9ufSBjYWxsYmFja1xuICogQHJldHVybiB7T2JqZWN0fVxuICovXG5mdW5jdGlvbiBsaXN0ZW5TZWxlY3RvcihzZWxlY3RvciwgdHlwZSwgY2FsbGJhY2spIHtcbiAgICByZXR1cm4gZGVsZWdhdGUoZG9jdW1lbnQuYm9keSwgc2VsZWN0b3IsIHR5cGUsIGNhbGxiYWNrKTtcbn1cblxubW9kdWxlLmV4cG9ydHMgPSBsaXN0ZW47XG5cblxuLyoqKi8gfSksXG5cbi8qKiovIDgxNzpcbi8qKiovIChmdW5jdGlvbihtb2R1bGUpIHtcblxuZnVuY3Rpb24gc2VsZWN0KGVsZW1lbnQpIHtcbiAgICB2YXIgc2VsZWN0ZWRUZXh0O1xuXG4gICAgaWYgKGVsZW1lbnQubm9kZU5hbWUgPT09ICdTRUxFQ1QnKSB7XG4gICAgICAgIGVsZW1lbnQuZm9jdXMoKTtcblxuICAgICAgICBzZWxlY3RlZFRleHQgPSBlbGVtZW50LnZhbHVlO1xuICAgIH1cbiAgICBlbHNlIGlmIChlbGVtZW50Lm5vZGVOYW1lID09PSAnSU5QVVQnIHx8IGVsZW1lbnQubm9kZU5hbWUgPT09ICdURVhUQVJFQScpIHtcbiAgICAgICAgdmFyIGlzUmVhZE9ubHkgPSBlbGVtZW50Lmhhc0F0dHJpYnV0ZSgncmVhZG9ubHknKTtcblxuICAgICAgICBpZiAoIWlzUmVhZE9ubHkpIHtcbiAgICAgICAgICAgIGVsZW1lbnQuc2V0QXR0cmlidXRlKCdyZWFkb25seScsICcnKTtcbiAgICAgICAgfVxuXG4gICAgICAgIGVsZW1lbnQuc2VsZWN0KCk7XG4gICAgICAgIGVsZW1lbnQuc2V0U2VsZWN0aW9uUmFuZ2UoMCwgZWxlbWVudC52YWx1ZS5sZW5ndGgpO1xuXG4gICAgICAgIGlmICghaXNSZWFkT25seSkge1xuICAgICAgICAgICAgZWxlbWVudC5yZW1vdmVBdHRyaWJ1dGUoJ3JlYWRvbmx5Jyk7XG4gICAgICAgIH1cblxuICAgICAgICBzZWxlY3RlZFRleHQgPSBlbGVtZW50LnZhbHVlO1xuICAgIH1cbiAgICBlbHNlIHtcbiAgICAgICAgaWYgKGVsZW1lbnQuaGFzQXR0cmlidXRlKCdjb250ZW50ZWRpdGFibGUnKSkge1xuICAgICAgICAgICAgZWxlbWVudC5mb2N1cygpO1xuICAgICAgICB9XG5cbiAgICAgICAgdmFyIHNlbGVjdGlvbiA9IHdpbmRvdy5nZXRTZWxlY3Rpb24oKTtcbiAgICAgICAgdmFyIHJhbmdlID0gZG9jdW1lbnQuY3JlYXRlUmFuZ2UoKTtcblxuICAgICAgICByYW5nZS5zZWxlY3ROb2RlQ29udGVudHMoZWxlbWVudCk7XG4gICAgICAgIHNlbGVjdGlvbi5yZW1vdmVBbGxSYW5nZXMoKTtcbiAgICAgICAgc2VsZWN0aW9uLmFkZFJhbmdlKHJhbmdlKTtcblxuICAgICAgICBzZWxlY3RlZFRleHQgPSBzZWxlY3Rpb24udG9TdHJpbmcoKTtcbiAgICB9XG5cbiAgICByZXR1cm4gc2VsZWN0ZWRUZXh0O1xufVxuXG5tb2R1bGUuZXhwb3J0cyA9IHNlbGVjdDtcblxuXG4vKioqLyB9KSxcblxuLyoqKi8gMjc5OlxuLyoqKi8gKGZ1bmN0aW9uKG1vZHVsZSkge1xuXG5mdW5jdGlvbiBFICgpIHtcbiAgLy8gS2VlcCB0aGlzIGVtcHR5IHNvIGl0J3MgZWFzaWVyIHRvIGluaGVyaXQgZnJvbVxuICAvLyAodmlhIGh0dHBzOi8vZ2l0aHViLmNvbS9saXBzbWFjayBmcm9tIGh0dHBzOi8vZ2l0aHViLmNvbS9zY290dGNvcmdhbi90aW55LWVtaXR0ZXIvaXNzdWVzLzMpXG59XG5cbkUucHJvdG90eXBlID0ge1xuICBvbjogZnVuY3Rpb24gKG5hbWUsIGNhbGxiYWNrLCBjdHgpIHtcbiAgICB2YXIgZSA9IHRoaXMuZSB8fCAodGhpcy5lID0ge30pO1xuXG4gICAgKGVbbmFtZV0gfHwgKGVbbmFtZV0gPSBbXSkpLnB1c2goe1xuICAgICAgZm46IGNhbGxiYWNrLFxuICAgICAgY3R4OiBjdHhcbiAgICB9KTtcblxuICAgIHJldHVybiB0aGlzO1xuICB9LFxuXG4gIG9uY2U6IGZ1bmN0aW9uIChuYW1lLCBjYWxsYmFjaywgY3R4KSB7XG4gICAgdmFyIHNlbGYgPSB0aGlzO1xuICAgIGZ1bmN0aW9uIGxpc3RlbmVyICgpIHtcbiAgICAgIHNlbGYub2ZmKG5hbWUsIGxpc3RlbmVyKTtcbiAgICAgIGNhbGxiYWNrLmFwcGx5KGN0eCwgYXJndW1lbnRzKTtcbiAgICB9O1xuXG4gICAgbGlzdGVuZXIuXyA9IGNhbGxiYWNrXG4gICAgcmV0dXJuIHRoaXMub24obmFtZSwgbGlzdGVuZXIsIGN0eCk7XG4gIH0sXG5cbiAgZW1pdDogZnVuY3Rpb24gKG5hbWUpIHtcbiAgICB2YXIgZGF0YSA9IFtdLnNsaWNlLmNhbGwoYXJndW1lbnRzLCAxKTtcbiAgICB2YXIgZXZ0QXJyID0gKCh0aGlzLmUgfHwgKHRoaXMuZSA9IHt9KSlbbmFtZV0gfHwgW10pLnNsaWNlKCk7XG4gICAgdmFyIGkgPSAwO1xuICAgIHZhciBsZW4gPSBldnRBcnIubGVuZ3RoO1xuXG4gICAgZm9yIChpOyBpIDwgbGVuOyBpKyspIHtcbiAgICAgIGV2dEFycltpXS5mbi5hcHBseShldnRBcnJbaV0uY3R4LCBkYXRhKTtcbiAgICB9XG5cbiAgICByZXR1cm4gdGhpcztcbiAgfSxcblxuICBvZmY6IGZ1bmN0aW9uIChuYW1lLCBjYWxsYmFjaykge1xuICAgIHZhciBlID0gdGhpcy5lIHx8ICh0aGlzLmUgPSB7fSk7XG4gICAgdmFyIGV2dHMgPSBlW25hbWVdO1xuICAgIHZhciBsaXZlRXZlbnRzID0gW107XG5cbiAgICBpZiAoZXZ0cyAmJiBjYWxsYmFjaykge1xuICAgICAgZm9yICh2YXIgaSA9IDAsIGxlbiA9IGV2dHMubGVuZ3RoOyBpIDwgbGVuOyBpKyspIHtcbiAgICAgICAgaWYgKGV2dHNbaV0uZm4gIT09IGNhbGxiYWNrICYmIGV2dHNbaV0uZm4uXyAhPT0gY2FsbGJhY2spXG4gICAgICAgICAgbGl2ZUV2ZW50cy5wdXNoKGV2dHNbaV0pO1xuICAgICAgfVxuICAgIH1cblxuICAgIC8vIFJlbW92ZSBldmVudCBmcm9tIHF1ZXVlIHRvIHByZXZlbnQgbWVtb3J5IGxlYWtcbiAgICAvLyBTdWdnZXN0ZWQgYnkgaHR0cHM6Ly9naXRodWIuY29tL2xhemRcbiAgICAvLyBSZWY6IGh0dHBzOi8vZ2l0aHViLmNvbS9zY290dGNvcmdhbi90aW55LWVtaXR0ZXIvY29tbWl0L2M2ZWJmYWE5YmM5NzNiMzNkMTEwYTg0YTMwNzc0MmI3Y2Y5NGM5NTMjY29tbWl0Y29tbWVudC01MDI0OTEwXG5cbiAgICAobGl2ZUV2ZW50cy5sZW5ndGgpXG4gICAgICA/IGVbbmFtZV0gPSBsaXZlRXZlbnRzXG4gICAgICA6IGRlbGV0ZSBlW25hbWVdO1xuXG4gICAgcmV0dXJuIHRoaXM7XG4gIH1cbn07XG5cbm1vZHVsZS5leHBvcnRzID0gRTtcbm1vZHVsZS5leHBvcnRzLlRpbnlFbWl0dGVyID0gRTtcblxuXG4vKioqLyB9KVxuXG4vKioqKioqLyBcdH0pO1xuLyoqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKi9cbi8qKioqKiovIFx0Ly8gVGhlIG1vZHVsZSBjYWNoZVxuLyoqKioqKi8gXHR2YXIgX193ZWJwYWNrX21vZHVsZV9jYWNoZV9fID0ge307XG4vKioqKioqLyBcdFxuLyoqKioqKi8gXHQvLyBUaGUgcmVxdWlyZSBmdW5jdGlvblxuLyoqKioqKi8gXHRmdW5jdGlvbiBfX3dlYnBhY2tfcmVxdWlyZV9fKG1vZHVsZUlkKSB7XG4vKioqKioqLyBcdFx0Ly8gQ2hlY2sgaWYgbW9kdWxlIGlzIGluIGNhY2hlXG4vKioqKioqLyBcdFx0aWYoX193ZWJwYWNrX21vZHVsZV9jYWNoZV9fW21vZHVsZUlkXSkge1xuLyoqKioqKi8gXHRcdFx0cmV0dXJuIF9fd2VicGFja19tb2R1bGVfY2FjaGVfX1ttb2R1bGVJZF0uZXhwb3J0cztcbi8qKioqKiovIFx0XHR9XG4vKioqKioqLyBcdFx0Ly8gQ3JlYXRlIGEgbmV3IG1vZHVsZSAoYW5kIHB1dCBpdCBpbnRvIHRoZSBjYWNoZSlcbi8qKioqKiovIFx0XHR2YXIgbW9kdWxlID0gX193ZWJwYWNrX21vZHVsZV9jYWNoZV9fW21vZHVsZUlkXSA9IHtcbi8qKioqKiovIFx0XHRcdC8vIG5vIG1vZHVsZS5pZCBuZWVkZWRcbi8qKioqKiovIFx0XHRcdC8vIG5vIG1vZHVsZS5sb2FkZWQgbmVlZGVkXG4vKioqKioqLyBcdFx0XHRleHBvcnRzOiB7fVxuLyoqKioqKi8gXHRcdH07XG4vKioqKioqLyBcdFxuLyoqKioqKi8gXHRcdC8vIEV4ZWN1dGUgdGhlIG1vZHVsZSBmdW5jdGlvblxuLyoqKioqKi8gXHRcdF9fd2VicGFja19tb2R1bGVzX19bbW9kdWxlSWRdKG1vZHVsZSwgbW9kdWxlLmV4cG9ydHMsIF9fd2VicGFja19yZXF1aXJlX18pO1xuLyoqKioqKi8gXHRcbi8qKioqKiovIFx0XHQvLyBSZXR1cm4gdGhlIGV4cG9ydHMgb2YgdGhlIG1vZHVsZVxuLyoqKioqKi8gXHRcdHJldHVybiBtb2R1bGUuZXhwb3J0cztcbi8qKioqKiovIFx0fVxuLyoqKioqKi8gXHRcbi8qKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKiovXG4vKioqKioqLyBcdC8qIHdlYnBhY2svcnVudGltZS9jb21wYXQgZ2V0IGRlZmF1bHQgZXhwb3J0ICovXG4vKioqKioqLyBcdCFmdW5jdGlvbigpIHtcbi8qKioqKiovIFx0XHQvLyBnZXREZWZhdWx0RXhwb3J0IGZ1bmN0aW9uIGZvciBjb21wYXRpYmlsaXR5IHdpdGggbm9uLWhhcm1vbnkgbW9kdWxlc1xuLyoqKioqKi8gXHRcdF9fd2VicGFja19yZXF1aXJlX18ubiA9IGZ1bmN0aW9uKG1vZHVsZSkge1xuLyoqKioqKi8gXHRcdFx0dmFyIGdldHRlciA9IG1vZHVsZSAmJiBtb2R1bGUuX19lc01vZHVsZSA/XG4vKioqKioqLyBcdFx0XHRcdGZ1bmN0aW9uKCkgeyByZXR1cm4gbW9kdWxlWydkZWZhdWx0J107IH0gOlxuLyoqKioqKi8gXHRcdFx0XHRmdW5jdGlvbigpIHsgcmV0dXJuIG1vZHVsZTsgfTtcbi8qKioqKiovIFx0XHRcdF9fd2VicGFja19yZXF1aXJlX18uZChnZXR0ZXIsIHsgYTogZ2V0dGVyIH0pO1xuLyoqKioqKi8gXHRcdFx0cmV0dXJuIGdldHRlcjtcbi8qKioqKiovIFx0XHR9O1xuLyoqKioqKi8gXHR9KCk7XG4vKioqKioqLyBcdFxuLyoqKioqKi8gXHQvKiB3ZWJwYWNrL3J1bnRpbWUvZGVmaW5lIHByb3BlcnR5IGdldHRlcnMgKi9cbi8qKioqKiovIFx0IWZ1bmN0aW9uKCkge1xuLyoqKioqKi8gXHRcdC8vIGRlZmluZSBnZXR0ZXIgZnVuY3Rpb25zIGZvciBoYXJtb255IGV4cG9ydHNcbi8qKioqKiovIFx0XHRfX3dlYnBhY2tfcmVxdWlyZV9fLmQgPSBmdW5jdGlvbihleHBvcnRzLCBkZWZpbml0aW9uKSB7XG4vKioqKioqLyBcdFx0XHRmb3IodmFyIGtleSBpbiBkZWZpbml0aW9uKSB7XG4vKioqKioqLyBcdFx0XHRcdGlmKF9fd2VicGFja19yZXF1aXJlX18ubyhkZWZpbml0aW9uLCBrZXkpICYmICFfX3dlYnBhY2tfcmVxdWlyZV9fLm8oZXhwb3J0cywga2V5KSkge1xuLyoqKioqKi8gXHRcdFx0XHRcdE9iamVjdC5kZWZpbmVQcm9wZXJ0eShleHBvcnRzLCBrZXksIHsgZW51bWVyYWJsZTogdHJ1ZSwgZ2V0OiBkZWZpbml0aW9uW2tleV0gfSk7XG4vKioqKioqLyBcdFx0XHRcdH1cbi8qKioqKiovIFx0XHRcdH1cbi8qKioqKiovIFx0XHR9O1xuLyoqKioqKi8gXHR9KCk7XG4vKioqKioqLyBcdFxuLyoqKioqKi8gXHQvKiB3ZWJwYWNrL3J1bnRpbWUvaGFzT3duUHJvcGVydHkgc2hvcnRoYW5kICovXG4vKioqKioqLyBcdCFmdW5jdGlvbigpIHtcbi8qKioqKiovIFx0XHRfX3dlYnBhY2tfcmVxdWlyZV9fLm8gPSBmdW5jdGlvbihvYmosIHByb3ApIHsgcmV0dXJuIE9iamVjdC5wcm90b3R5cGUuaGFzT3duUHJvcGVydHkuY2FsbChvYmosIHByb3ApOyB9XG4vKioqKioqLyBcdH0oKTtcbi8qKioqKiovIFx0XG4vKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqL1xuLyoqKioqKi8gXHQvLyBtb2R1bGUgZXhwb3J0cyBtdXN0IGJlIHJldHVybmVkIGZyb20gcnVudGltZSBzbyBlbnRyeSBpbmxpbmluZyBpcyBkaXNhYmxlZFxuLyoqKioqKi8gXHQvLyBzdGFydHVwXG4vKioqKioqLyBcdC8vIExvYWQgZW50cnkgbW9kdWxlIGFuZCByZXR1cm4gZXhwb3J0c1xuLyoqKioqKi8gXHRyZXR1cm4gX193ZWJwYWNrX3JlcXVpcmVfXyg2ODYpO1xuLyoqKioqKi8gfSkoKVxuLmRlZmF1bHQ7XG59KTsiLCAiLy8gUHV0IHlvdXIgY3VzdG9tIEpTIGNvZGUgaGVyZVxuY29uc3QgZ2V0VGhlbWUgPSAoKSA9PiBkb2N1bWVudC5xdWVyeVNlbGVjdG9yKCdodG1sJykuZ2V0QXR0cmlidXRlKCdkYXRhLWJzLXRoZW1lJyk7XG5cbmNvbnN0IHVwZGF0ZUxvZ29zID0gKCkgPT4ge1xuICAgIGNvbnN0IHRoZW1lID0gZ2V0VGhlbWUoKTtcbiAgICBsb2dvcyA9IHtcbiAgICAgICAgaGY6IHtcbiAgICAgICAgICAgIGxpZ2h0OiAnYXNzZXRzL2ltYWdlcy9sb2dvLWhmLWxpZ2h0LnBuZycsXG4gICAgICAgICAgICBkYXJrOiAnYXNzZXRzL2ltYWdlcy9sb2dvLWhmLWRhcmsucG5nJ1xuICAgICAgICB9LFxuICAgICAgICBlbnRhbHBpYzoge1xuICAgICAgICAgICAgbGlnaHQ6ICdhc3NldHMvaW1hZ2VzL2xvZ28tZW50YWxwaWMtbGlnaHQuanBnJyxcbiAgICAgICAgICAgIGRhcms6ICdhc3NldHMvaW1hZ2VzL2xvZ28tZW50YWxwaWMtZGFyay5wbmcnXG4gICAgICAgIH1cbiAgICB9O1xuICAgIGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdoZi1sb2dvJykuc3JjID0gbG9nb3MuaGZbdGhlbWVdO1xuICAgIGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdlbnRhbHBpYy1sb2dvJykuc3JjID0gbG9nb3MuZW50YWxwaWNbdGhlbWVdO1xufTtcblxuKGFzeW5jICgpID0+IHt9KSgpO1xuIiwgImZ1bmN0aW9uIGUoZSl7cmV0dXJuIG5ldyBQcm9taXNlKGZ1bmN0aW9uKG4scix0KXsodD1uZXcgWE1MSHR0cFJlcXVlc3QpLm9wZW4oXCJHRVRcIixlLHQud2l0aENyZWRlbnRpYWxzPSEwKSx0Lm9ubG9hZD1mdW5jdGlvbigpezIwMD09PXQuc3RhdHVzP24oKTpyKCl9LHQuc2VuZCgpfSl9dmFyIG4scj0obj1kb2N1bWVudC5jcmVhdGVFbGVtZW50KFwibGlua1wiKSkucmVsTGlzdCYmbi5yZWxMaXN0LnN1cHBvcnRzJiZuLnJlbExpc3Quc3VwcG9ydHMoXCJwcmVmZXRjaFwiKT9mdW5jdGlvbihlKXtyZXR1cm4gbmV3IFByb21pc2UoZnVuY3Rpb24obixyLHQpeyh0PWRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoXCJsaW5rXCIpKS5yZWw9XCJwcmVmZXRjaFwiLHQuaHJlZj1lLHQub25sb2FkPW4sdC5vbmVycm9yPXIsZG9jdW1lbnQuaGVhZC5hcHBlbmRDaGlsZCh0KX0pfTplLHQ9d2luZG93LnJlcXVlc3RJZGxlQ2FsbGJhY2t8fGZ1bmN0aW9uKGUpe3ZhciBuPURhdGUubm93KCk7cmV0dXJuIHNldFRpbWVvdXQoZnVuY3Rpb24oKXtlKHtkaWRUaW1lb3V0OiExLHRpbWVSZW1haW5pbmc6ZnVuY3Rpb24oKXtyZXR1cm4gTWF0aC5tYXgoMCw1MC0oRGF0ZS5ub3coKS1uKSl9fSl9LDEpfSxvPW5ldyBTZXQsaT1uZXcgU2V0LGM9ITE7ZnVuY3Rpb24gYShlKXtpZihlKXtpZihlLnNhdmVEYXRhKXJldHVybiBuZXcgRXJyb3IoXCJTYXZlLURhdGEgaXMgZW5hYmxlZFwiKTtpZigvMmcvLnRlc3QoZS5lZmZlY3RpdmVUeXBlKSlyZXR1cm4gbmV3IEVycm9yKFwibmV0d29yayBjb25kaXRpb25zIGFyZSBwb29yXCIpfXJldHVybiEwfWZ1bmN0aW9uIHUoZSl7aWYoZXx8KGU9e30pLHdpbmRvdy5JbnRlcnNlY3Rpb25PYnNlcnZlcil7dmFyIG49ZnVuY3Rpb24oZSl7ZT1lfHwxO3ZhciBuPVtdLHI9MDtmdW5jdGlvbiB0KCl7cjxlJiZuLmxlbmd0aD4wJiYobi5zaGlmdCgpKCkscisrKX1yZXR1cm5bZnVuY3Rpb24oZSl7bi5wdXNoKGUpPjF8fHQoKX0sZnVuY3Rpb24oKXtyLS0sdCgpfV19KGUudGhyb3R0bGV8fDEvMCkscj1uWzBdLGE9blsxXSx1PWUubGltaXR8fDEvMCxsPWUub3JpZ2luc3x8W2xvY2F0aW9uLmhvc3RuYW1lXSxkPWUuaWdub3Jlc3x8W10saD1lLmRlbGF5fHwwLHA9W10sbT1lLnRpbWVvdXRGbnx8dCx3PVwiZnVuY3Rpb25cIj09dHlwZW9mIGUuaHJlZkZuJiZlLmhyZWZGbixnPWUucHJlcmVuZGVyfHwhMTtjPWUucHJlcmVuZGVyQW5kUHJlZmV0Y2h8fCExO3ZhciB2PW5ldyBJbnRlcnNlY3Rpb25PYnNlcnZlcihmdW5jdGlvbihuKXtuLmZvckVhY2goZnVuY3Rpb24obil7aWYobi5pc0ludGVyc2VjdGluZylwLnB1c2goKG49bi50YXJnZXQpLmhyZWYpLGZ1bmN0aW9uKGUsbil7bj9zZXRUaW1lb3V0KGUsbik6ZSgpfShmdW5jdGlvbigpey0xIT09cC5pbmRleE9mKG4uaHJlZikmJih2LnVub2JzZXJ2ZShuKSwoY3x8ZykmJmkuc2l6ZTwxP2Yodz93KG4pOm4uaHJlZikuY2F0Y2goZnVuY3Rpb24obil7aWYoIWUub25FcnJvcil0aHJvdyBuO2Uub25FcnJvcihuKX0pOm8uc2l6ZTx1JiYhZyYmcihmdW5jdGlvbigpe3Modz93KG4pOm4uaHJlZixlLnByaW9yaXR5KS50aGVuKGEpLmNhdGNoKGZ1bmN0aW9uKG4pe2EoKSxlLm9uRXJyb3ImJmUub25FcnJvcihuKX0pfSkpfSxoKTtlbHNle3ZhciB0PXAuaW5kZXhPZigobj1uLnRhcmdldCkuaHJlZik7dD4tMSYmcC5zcGxpY2UodCl9fSl9LHt0aHJlc2hvbGQ6ZS50aHJlc2hvbGR8fDB9KTtyZXR1cm4gbShmdW5jdGlvbigpeyhlLmVsfHxkb2N1bWVudCkucXVlcnlTZWxlY3RvckFsbChcImFcIikuZm9yRWFjaChmdW5jdGlvbihlKXtsLmxlbmd0aCYmIWwuaW5jbHVkZXMoZS5ob3N0bmFtZSl8fGZ1bmN0aW9uIGUobixyKXtyZXR1cm4gQXJyYXkuaXNBcnJheShyKT9yLnNvbWUoZnVuY3Rpb24ocil7cmV0dXJuIGUobixyKX0pOihyLnRlc3R8fHIpLmNhbGwocixuLmhyZWYsbil9KGUsZCl8fHYub2JzZXJ2ZShlKX0pfSx7dGltZW91dDplLnRpbWVvdXR8fDJlM30pLGZ1bmN0aW9uKCl7by5jbGVhcigpLHYuZGlzY29ubmVjdCgpfX19ZnVuY3Rpb24gcyhuLHQsdSl7dmFyIHM9YShuYXZpZ2F0b3IuY29ubmVjdGlvbik7cmV0dXJuIHMgaW5zdGFuY2VvZiBFcnJvcj9Qcm9taXNlLnJlamVjdChuZXcgRXJyb3IoXCJDYW5ub3QgcHJlZmV0Y2gsIFwiK3MubWVzc2FnZSkpOihpLnNpemU+MCYmIWMmJmNvbnNvbGUud2FybihcIltXYXJuaW5nXSBZb3UgYXJlIHVzaW5nIGJvdGggcHJlZmV0Y2hpbmcgYW5kIHByZXJlbmRlcmluZyBvbiB0aGUgc2FtZSBkb2N1bWVudFwiKSxQcm9taXNlLmFsbChbXS5jb25jYXQobikubWFwKGZ1bmN0aW9uKG4pe2lmKCFvLmhhcyhuKSlyZXR1cm4gby5hZGQobiksKHQ/ZnVuY3Rpb24obil7cmV0dXJuIHdpbmRvdy5mZXRjaD9mZXRjaChuLHtjcmVkZW50aWFsczpcImluY2x1ZGVcIn0pOmUobil9OnIpKG5ldyBVUkwobixsb2NhdGlvbi5ocmVmKS50b1N0cmluZygpKX0pKSl9ZnVuY3Rpb24gZihlLG4pe3ZhciByPWEobmF2aWdhdG9yLmNvbm5lY3Rpb24pO2lmKHIgaW5zdGFuY2VvZiBFcnJvcilyZXR1cm4gUHJvbWlzZS5yZWplY3QobmV3IEVycm9yKFwiQ2Fubm90IHByZXJlbmRlciwgXCIrci5tZXNzYWdlKSk7aWYoIUhUTUxTY3JpcHRFbGVtZW50LnN1cHBvcnRzKFwic3BlY3VsYXRpb25ydWxlc1wiKSlyZXR1cm4gcyhlKSxQcm9taXNlLnJlamVjdChuZXcgRXJyb3IoXCJUaGlzIGJyb3dzZXIgZG9lcyBub3Qgc3VwcG9ydCB0aGUgc3BlY3VsYXRpb24gcnVsZXMgQVBJLiBGYWxsaW5nIGJhY2sgdG8gcHJlZmV0Y2guXCIpKTtpZihkb2N1bWVudC5xdWVyeVNlbGVjdG9yKCdzY3JpcHRbdHlwZT1cInNwZWN1bGF0aW9ucnVsZXNcIl0nKSlyZXR1cm4gUHJvbWlzZS5yZWplY3QobmV3IEVycm9yKFwiU3BlY3VsYXRpb24gUnVsZXMgaXMgYWxyZWFkeSBkZWZpbmVkIGFuZCBjYW5ub3QgYmUgYWx0ZXJlZC5cIikpO2Zvcih2YXIgdD0wLHU9W10uY29uY2F0KGUpO3Q8dS5sZW5ndGg7dCs9MSl7dmFyIGY9dVt0XTtpZih3aW5kb3cubG9jYXRpb24ub3JpZ2luIT09bmV3IFVSTChmLHdpbmRvdy5sb2NhdGlvbi5ocmVmKS5vcmlnaW4pcmV0dXJuIFByb21pc2UucmVqZWN0KG5ldyBFcnJvcihcIk9ubHkgc2FtZSBvcmlnaW4gVVJMcyBhcmUgYWxsb3dlZDogXCIrZikpO2kuYWRkKGYpfW8uc2l6ZT4wJiYhYyYmY29uc29sZS53YXJuKFwiW1dhcm5pbmddIFlvdSBhcmUgdXNpbmcgYm90aCBwcmVmZXRjaGluZyBhbmQgcHJlcmVuZGVyaW5nIG9uIHRoZSBzYW1lIGRvY3VtZW50XCIpO3ZhciBsPWZ1bmN0aW9uKGUpe3ZhciBuPWRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoXCJzY3JpcHRcIik7bi50eXBlPVwic3BlY3VsYXRpb25ydWxlc1wiLG4udGV4dD0ne1wicHJlcmVuZGVyXCI6W3tcInNvdXJjZVwiOiBcImxpc3RcIixcInVybHNcIjogW1wiJytBcnJheS5mcm9tKGUpLmpvaW4oJ1wiLFwiJykrJ1wiXX1dfSc7dHJ5e2RvY3VtZW50LmhlYWQuYXBwZW5kQ2hpbGQobil9Y2F0Y2goZSl7cmV0dXJuIGV9cmV0dXJuITB9KGkpO3JldHVybiEwPT09bD9Qcm9taXNlLnJlc29sdmUoKTpQcm9taXNlLnJlamVjdChsKX1leHBvcnR7dSBhcyBsaXN0ZW4scyBhcyBwcmVmZXRjaCxmIGFzIHByZXJlbmRlcn07XG4iLCAiLy8gaHR0cHM6Ly9naXRodWIuY29tL0dvb2dsZUNocm9tZUxhYnMvcXVpY2tsaW5rXHJcbmltcG9ydCB7IGxpc3RlbiB9IGZyb20gJ3F1aWNrbGluay9kaXN0L3F1aWNrbGluay5tanMnO1xyXG5saXN0ZW4oe1xyXG4gICAgaWdub3JlczogW1xyXG4gICAgICAgIC9cXC9hcGlcXC8/LyxcclxuICAgICAgICB1cmkgPT4gdXJpLmluY2x1ZGVzKCcuemlwJyksXHJcbiAgICAgICAgKHVyaSwgZWxlbSkgPT4gZWxlbS5oYXNBdHRyaWJ1dGUoJ25vcHJlZmV0Y2gnKSxcclxuICAgICAgICAodXJpLCBlbGVtKSA9PiBlbGVtLmhhc2ggJiYgZWxlbS5wYXRobmFtZSA9PT0gd2luZG93LmxvY2F0aW9uLnBhdGhuYW1lLFxyXG4gICAgXVxyXG59KTtcclxuXHJcbi8vIGh0dHBzOi8vZ2l0aHViLmNvbS9hRmFya2FzL2xhenlzaXplcy90cmVlL2doLXBhZ2VzL3BsdWdpbnMvbmF0aXZlLWxvYWRpbmdcclxuaW1wb3J0IGxhenlTaXplcyBmcm9tICdsYXp5c2l6ZXMnO1xyXG5pbXBvcnQgJ2xhenlzaXplcy9wbHVnaW5zL25hdGl2ZS1sb2FkaW5nL2xzLm5hdGl2ZS1sb2FkaW5nJztcclxuXHJcbmxhenlTaXplcy5jZmcubmF0aXZlTG9hZGluZyA9IHtcclxuICAgIHNldExvYWRpbmdBdHRyaWJ1dGU6IHRydWUsXHJcbiAgICBkaXNhYmxlTGlzdGVuZXJzOiB7XHJcbiAgICAgICAgc2Nyb2xsOiB0cnVlXHJcbiAgICB9XHJcbn07XHJcbiIsICIvKiFcclxuICogY2xpcGJvYXJkLmpzIGZvciBCb290c3RyYXAgYmFzZWQgVGh1bGl0ZSBzaXRlc1xyXG4gKiBDb3B5cmlnaHQgMjAyMS0yMDI0IFRodWxpdGVcclxuICogTGljZW5zZWQgdW5kZXIgdGhlIE1JVCBMaWNlbnNlXHJcbiAqL1xyXG5cclxuaW1wb3J0IENsaXBib2FyZCBmcm9tICdjbGlwYm9hcmQnO1xyXG5cclxuKCgpID0+IHtcclxuICAgICd1c2Ugc3RyaWN0JztcclxuXHJcbiAgICB2YXIgY2IgPSBkb2N1bWVudC5nZXRFbGVtZW50c0J5Q2xhc3NOYW1lKCdoaWdobGlnaHQnKTtcclxuXHJcbiAgICBmb3IgKHZhciBpID0gMDsgaSA8IGNiLmxlbmd0aDsgKytpKSB7XHJcbiAgICAgICAgdmFyIGVsZW1lbnQgPSBjYltpXTtcclxuICAgICAgICBlbGVtZW50Lmluc2VydEFkamFjZW50SFRNTCgnYWZ0ZXJiZWdpbicsICc8ZGl2IGNsYXNzPVwiY29weVwiPjxidXR0b24gdGl0bGU9XCJDb3B5IHRvIGNsaXBib2FyZFwiIGNsYXNzPVwiYnRuLWNvcHlcIiBhcmlhLWxhYmVsPVwiQ2xpcGJvYXJkIGJ1dHRvblwiPjxkaXY+PC9kaXY+PC9idXR0b24+PC9kaXY+Jyk7XHJcbiAgICB9XHJcblxyXG4gICAgdmFyIGNsaXBib2FyZCA9IG5ldyBDbGlwYm9hcmQoJy5idG4tY29weScsIHtcclxuICAgICAgICB0YXJnZXQ6IGZ1bmN0aW9uICh0cmlnZ2VyKSB7XHJcbiAgICAgICAgICAgIHJldHVybiB0cmlnZ2VyLnBhcmVudE5vZGUubmV4dEVsZW1lbnRTaWJsaW5nO1xyXG4gICAgICAgIH1cclxuICAgIH0pO1xyXG5cclxuICAgIGNsaXBib2FyZC5vbignc3VjY2VzcycsIGZ1bmN0aW9uIChlKSB7XHJcbiAgICAgICAgLypcclxuICAgICAgY29uc29sZS5pbmZvKCdBY3Rpb246JywgZS5hY3Rpb24pO1xyXG4gICAgICBjb25zb2xlLmluZm8oJ1RleHQ6JywgZS50ZXh0KTtcclxuICAgICAgY29uc29sZS5pbmZvKCdUcmlnZ2VyOicsIGUudHJpZ2dlcik7XHJcbiAgICAgICovXHJcblxyXG4gICAgICAgIGUuY2xlYXJTZWxlY3Rpb24oKTtcclxuICAgIH0pO1xyXG5cclxuICAgIGNsaXBib2FyZC5vbignZXJyb3InLCBmdW5jdGlvbiAoZSkge1xyXG4gICAgICAgIGNvbnNvbGUuZXJyb3IoJ0FjdGlvbjonLCBlLmFjdGlvbik7XHJcbiAgICAgICAgY29uc29sZS5lcnJvcignVHJpZ2dlcjonLCBlLnRyaWdnZXIpO1xyXG4gICAgfSk7XHJcbn0pKCk7XHJcbiIsICJjb25zdCB0b3BCdXR0b24gPSBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgndG9Ub3AnKTtcclxuXHJcbmlmICh0b3BCdXR0b24gIT09IG51bGwpIHtcclxuICAgIHRvcEJ1dHRvbi5jbGFzc0xpc3QucmVtb3ZlKCdmYWRlJyk7XHJcbiAgICB3aW5kb3cub25zY3JvbGwgPSBmdW5jdGlvbiAoKSB7XHJcbiAgICAgICAgc2Nyb2xsRnVuY3Rpb24oKTtcclxuICAgIH07XHJcblxyXG4gICAgdG9wQnV0dG9uLmFkZEV2ZW50TGlzdGVuZXIoJ2NsaWNrJywgdG9wRnVuY3Rpb24pO1xyXG59XHJcblxyXG5mdW5jdGlvbiBzY3JvbGxGdW5jdGlvbigpIHtcclxuICAgIGlmIChkb2N1bWVudC5ib2R5LnNjcm9sbFRvcCA+IDI3MCB8fCBkb2N1bWVudC5kb2N1bWVudEVsZW1lbnQuc2Nyb2xsVG9wID4gMjcwKSB7XHJcbiAgICAgICAgdG9wQnV0dG9uLmNsYXNzTGlzdC5hZGQoJ2ZhZGUnKTtcclxuICAgIH0gZWxzZSB7XHJcbiAgICAgICAgdG9wQnV0dG9uLmNsYXNzTGlzdC5yZW1vdmUoJ2ZhZGUnKTtcclxuICAgIH1cclxufVxyXG5cclxuZnVuY3Rpb24gdG9wRnVuY3Rpb24oKSB7XHJcbiAgICBkb2N1bWVudC5ib2R5LnNjcm9sbFRvcCA9IDA7XHJcbiAgICBkb2N1bWVudC5kb2N1bWVudEVsZW1lbnQuc2Nyb2xsVG9wID0gMDtcclxufVxyXG4iLCAiLy8gQmFzZWQgb246IGh0dHBzOi8vZ2l0aHViLmNvbS9nb2h1Z29pby9odWdvRG9jcy9ibG9iL21hc3Rlci9fdmVuZG9yL2dpdGh1Yi5jb20vZ29odWdvaW8vZ29odWdvaW9UaGVtZS9hc3NldHMvanMvdGFicy5qc1xyXG5cclxuLyoqXHJcbiAqIFNjcmlwdHMgd2hpY2ggbWFuYWdlcyBDb2RlIFRvZ2dsZSB0YWJzLlxyXG4gKi9cclxudmFyIGk7XHJcbi8vIHN0b3JlIHRhYnMgdmFyaWFibGVcclxudmFyIGFsbFRhYnMgPSBkb2N1bWVudC5xdWVyeVNlbGVjdG9yQWxsKCdbZGF0YS10b2dnbGUtdGFiXScpO1xyXG52YXIgYWxsUGFuZXMgPSBkb2N1bWVudC5xdWVyeVNlbGVjdG9yQWxsKCdbZGF0YS1wYW5lXScpO1xyXG5cclxuZnVuY3Rpb24gdG9nZ2xlVGFicyhldmVudCkge1xyXG4gICAgaWYgKGV2ZW50LnRhcmdldCkge1xyXG4gICAgICAgIGV2ZW50LnByZXZlbnREZWZhdWx0KCk7XHJcbiAgICAgICAgdmFyIGNsaWNrZWRUYWIgPSBldmVudC5jdXJyZW50VGFyZ2V0O1xyXG4gICAgICAgIHZhciB0YXJnZXRLZXkgPSBjbGlja2VkVGFiLmdldEF0dHJpYnV0ZSgnZGF0YS10b2dnbGUtdGFiJyk7XHJcbiAgICB9IGVsc2Uge1xyXG4gICAgICAgIHZhciB0YXJnZXRLZXkgPSBldmVudDtcclxuICAgIH1cclxuICAgIC8vIFdlIHN0b3JlIHRoZSBjb25maWcgbGFuZ3VhZ2Ugc2VsZWN0ZWQgaW4gdXNlcnMnIGxvY2FsU3RvcmFnZVxyXG4gICAgaWYgKHdpbmRvdy5sb2NhbFN0b3JhZ2UpIHtcclxuICAgICAgICB3aW5kb3cubG9jYWxTdG9yYWdlLnNldEl0ZW0oJ2NvbmZpZ0xhbmdQcmVmJywgdGFyZ2V0S2V5KTtcclxuICAgIH1cclxuICAgIHZhciBzZWxlY3RlZFRhYnMgPSBkb2N1bWVudC5xdWVyeVNlbGVjdG9yQWxsKCdbZGF0YS10b2dnbGUtdGFiPScgKyB0YXJnZXRLZXkgKyAnXScpO1xyXG4gICAgdmFyIHNlbGVjdGVkUGFuZXMgPSBkb2N1bWVudC5xdWVyeVNlbGVjdG9yQWxsKCdbZGF0YS1wYW5lPScgKyB0YXJnZXRLZXkgKyAnXScpO1xyXG5cclxuICAgIGZvciAodmFyIGkgPSAwOyBpIDwgYWxsVGFicy5sZW5ndGg7IGkrKykge1xyXG4gICAgICAgIGFsbFRhYnNbaV0uY2xhc3NMaXN0LnJlbW92ZSgnYWN0aXZlJyk7XHJcbiAgICAgICAgYWxsUGFuZXNbaV0uY2xhc3NMaXN0LnJlbW92ZSgnYWN0aXZlJyk7XHJcbiAgICB9XHJcblxyXG4gICAgZm9yICh2YXIgaSA9IDA7IGkgPCBzZWxlY3RlZFRhYnMubGVuZ3RoOyBpKyspIHtcclxuICAgICAgICBzZWxlY3RlZFRhYnNbaV0uY2xhc3NMaXN0LmFkZCgnYWN0aXZlJyk7XHJcbiAgICAgICAgc2VsZWN0ZWRQYW5lc1tpXS5jbGFzc0xpc3QuYWRkKCdzaG93JywgJ2FjdGl2ZScpO1xyXG4gICAgfVxyXG59XHJcblxyXG5mb3IgKGkgPSAwOyBpIDwgYWxsVGFicy5sZW5ndGg7IGkrKykge1xyXG4gICAgYWxsVGFic1tpXS5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsIHRvZ2dsZVRhYnMpO1xyXG59XHJcbi8vIFVwb24gcGFnZSBsb2FkLCBpZiB1c2VyIGhhcyBhIHByZWZlcnJlZCBsYW5ndWFnZSBpbiBpdHMgbG9jYWxTdG9yYWdlLCB0YWJzIGFyZSBzZXQgdG8gaXQuXHJcbmlmICh3aW5kb3cubG9jYWxTdG9yYWdlLmdldEl0ZW0oJ2NvbmZpZ0xhbmdQcmVmJykpIHtcclxuICAgIHRvZ2dsZVRhYnMod2luZG93LmxvY2FsU3RvcmFnZS5nZXRJdGVtKCdjb25maWdMYW5nUHJlZicpKTtcclxufVxyXG4iLCAiLy8gYXBwLmpzIGZvciBjb25jYXRlbmF0aW9uIG9mIHNtYWxsZXIgbGlicmFyaWVzIHRvIHJlZHVjZSBIVFRQIHJlcXVlc3RzXHJcbid1c2Ugc3RyaWN0JztcclxuXHJcbi8vIGltcG9ydCBjb3JlLmpzXHJcbmltcG9ydCAnQHRodWxpdGUvY29yZS9hc3NldHMvanMvY29yZS5qcyc7XHJcblxyXG4vLyBJbXBvcnQgY2xpcGJvYXJkLmpzXHJcbmltcG9ydCAnanMvY2xpcGJvYXJkLmpzJztcclxuXHJcbi8vIEltcG9ydCB0by10b3AuanNcclxuaW1wb3J0ICdqcy90by10b3AuanMnO1xyXG5cclxuLy8gSW1wb3J0IHRhYnMuanNcclxuaW1wb3J0ICdqcy90YWJzLmpzJztcclxuXHJcbi8vIEltcG9ydCBjdXN0b20uanNcclxuaW1wb3J0ICdqcy9jdXN0b20uanMnO1xyXG5cclxuLypcclxuaW1wb3J0ICogYXMgcGFyYW1zIGZyb20gJ0BwYXJhbXMnO1xyXG5cclxuaWYgKHBhcmFtcy5lbnYgPT09ICdkZXZlbG9wbWVudCcpIHtcclxuICBjb25zb2xlLmxvZygnSHVnbyBkZXZlbG9wbWVudCBlbnZpcm9ubWVudCcpXHJcbn0gZWxzZSB7XHJcbiAgY29uc29sZS5sb2coJ0h1Z28gcHJvZHVjdGlvbiBlbnZpcm9ubWVudCcpXHJcbn1cclxuKi9cclxuIl0sCiAgIm1hcHBpbmdzIjogIjs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUFBO0FBQUE7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QUFXLE9BQU8sYUFBYSxRQUFRLGdCQUFnQixDQUFDO0FBQUEsRUFDNUQ7OztBQzFCQSxzQkFBTzsiLAogICJuYW1lcyI6IFsid2luZG93IiwgImxhenlTaXplcyIsICJkb2N1bWVudCIsICJEYXRlIiwgInNldFRpbWVvdXQiLCAiZSIsICJpIiwgImxvYWRNb2RlIiwgIndpbmRvdyIsICJkb2N1bWVudCIsICJsYXp5U2l6ZXMiLCAiZSIsICJfX3dlYnBhY2tfcmVxdWlyZV9fIiwgIkNsaXBib2FyZEFjdGlvbkN1dCIsICJmYWtlQ29weUFjdGlvbiIsICJDbGlwYm9hcmRBY3Rpb25Db3B5IiwgIl90eXBlb2YiLCAib2JqIiwgIkNsaXBib2FyZEFjdGlvbkRlZmF1bHQiLCAiaSIsICJvIiwgIl9zZXRQcm90b3R5cGVPZiIsICJwIiwgImUiLCAiX2dldFByb3RvdHlwZU9mIiwgIkNsaXBib2FyZCIsICJhY3Rpb24iLCAibW9kdWxlIiwgImV4cG9ydHMiLCAiZSIsICJuIiwgInIiLCAidCIsICJhIiwgInUiLCAicyIsICJmIiwgImxhenlTaXplcyIsICJpIiwgIkNsaXBib2FyZCIsICJlIiwgImkiXQp9Cg==
