(() => {
  const snap = document.querySelector(".snap");
  const nav = document.querySelector(".nav");
  const features = document.querySelector("#features");
  const panels = [...document.querySelectorAll(".feature-panel")];
  const hero = document.querySelector("#hero");

  const scrollToId = (id) => {
    const el = document.getElementById(id.replace("#", ""));
    if (!el || !snap) return;
    snap.scrollTo({ top: el.offsetTop, behavior: "smooth" });
  };

  document.querySelectorAll("[data-scroll]").forEach((link) => {
    link.addEventListener("click", (event) => {
      const href = link.getAttribute("href");
      if (!href || !href.startsWith("#")) return;
      event.preventDefault();
      scrollToId(href);
    });
  });

  const auroraRoot = document.getElementById("hero-aurora");
  if (auroraRoot && window.mountAurora) {
    window.mountAurora(auroraRoot, {
      colorStops: ["#3a1cff", "#ff6bb5", "#4cc2e9"],
      amplitude: 1.05,
      blend: 0.55,
      speed: 0.85,
      height: 0.55,
      glow: 0.85,
      brightness: 1.05,
      layers: 2,
      direction: "top",
      mouseInteraction: true,
      opacity: 0.95,
    });
  }

  const orbRoot = document.getElementById("hero-orb");
  if (orbRoot && window.mountOrb) {
    window.mountOrb(orbRoot, {
      colors: ["#9c43fe", "#4cc2e9", "#101499"],
      backgroundColor: "#05070c",
      size: 1.05,
      glow: 1.15,
      bleed: 1.1,
      hoverIntensity: 0.9,
      followCursor: true,
      rotateOnHover: true,
    });
  }

  const galaxyRoot = document.getElementById("galaxy");
  if (galaxyRoot && window.mountGalaxy) {
    window.mountGalaxy(galaxyRoot, {
      hueShift: 155,
      saturation: 0.18,
      starSpeed: 0.42,
      glowIntensity: 0.38,
      transparent: false,
      lightMode: false,
      backgroundColor: "#05070c",
    });
  }

  let featureIndex = 0;
  let featureTimer = 0;
  const FEATURE_MS = 6500;

  const setFeature = (index) => {
    featureIndex = (index + panels.length) % panels.length;
    panels.forEach((panel, i) => panel.classList.toggle("is-active", i === featureIndex));
    features?.classList.toggle("is-explain", featureIndex === 2);
    if (window.galaxySettings) {
      window.galaxySettings.hueShift = featureIndex === 2 ? 195 : 155;
      window.galaxySettings.saturation = featureIndex === 2 ? 0.55 : 0.18;
      window.galaxySettings.glowIntensity = featureIndex === 2 ? 0.55 : 0.38;
      window.galaxySettings.transparent = false;
      window.galaxySettings.lightMode = false;
      window.galaxyWake?.();
    }
  };

  const stopAuto = () => {
    window.clearInterval(featureTimer);
    featureTimer = 0;
  };

  const startAuto = () => {
    stopAuto();
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    featureTimer = window.setInterval(() => setFeature(featureIndex + 1), FEATURE_MS);
  };

  if (panels.length) {
    setFeature(0);
    startAuto();

    features?.addEventListener("click", () => {
      setFeature(featureIndex + 1);
      startAuto();
    });

    features?.addEventListener("pointerenter", stopAuto);
    features?.addEventListener("pointerleave", startAuto);
  }

  // Entire landing is dark — keep nav on the dark variant.
  nav?.classList.add("is-dark");

  document.querySelectorAll(".faq-item").forEach((item) => {
    const button = item.querySelector(".faq-q");
    button?.addEventListener("click", () => {
      const open = item.classList.contains("is-open");
      document.querySelectorAll(".faq-item").forEach((other) => {
        other.classList.remove("is-open");
        other.querySelector(".faq-q")?.setAttribute("aria-expanded", "false");
      });
      if (!open) {
        item.classList.add("is-open");
        button.setAttribute("aria-expanded", "true");
      }
    });
  });

  const fan = document.querySelector(".fan");
  if (fan) {
    const cards = [...fan.querySelectorAll(".fan-card")];
    fan.addEventListener("pointermove", (event) => {
      const rect = fan.getBoundingClientRect();
      const x = (event.clientX - rect.left) / rect.width - 0.5;
      cards.forEach((card, i) => {
        const mid = (cards.length - 1) / 2;
        const drift = (i - mid) * 6;
        card.style.transform = `rotate(${drift + x * 8}deg) translateY(${Math.abs(x) * -12}px)`;
      });
    });
    fan.addEventListener("pointerleave", () => {
      cards.forEach((card) => {
        card.style.transform = "";
      });
    });
  }
})();
