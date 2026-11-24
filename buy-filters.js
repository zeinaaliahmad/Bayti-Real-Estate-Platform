function toggleMobileMenu() {
  const menu = document.getElementById("mobileMenu");
  if (!menu) return;

  if (menu.className.indexOf("w3-show") === -1) {
    menu.className += " w3-show";
  } else {
    menu.className = menu.className.replace(" w3-show", "");
  }
}

document.addEventListener("DOMContentLoaded", function () {
  const dropdownButtons = document.querySelectorAll(".dropdown-toggle");
  const dropdownPanels = document.querySelectorAll(".dropdown-panel");
  const doneButtons = document.querySelectorAll(".done-btn");

  dropdownButtons.forEach(function (button) {
    button.addEventListener("click", function (e) {
      e.stopPropagation();

      const targetId = this.getAttribute("data-target");
      const targetPanel = document.getElementById(targetId);

      if (!targetPanel) return;

      dropdownPanels.forEach(function (panel) {
        if (panel !== targetPanel) {
          panel.classList.remove("open");
        }
      });

      targetPanel.classList.toggle("open");
    });
  });

  doneButtons.forEach(function (button) {
    button.addEventListener("click", function (e) {
      e.preventDefault();
      e.stopPropagation();

      const panelId = this.getAttribute("data-close");
      const panel = document.getElementById(panelId);

      if (panel) {
        panel.classList.remove("open");
      }
    });
  });

  document.addEventListener("click", function (e) {
    if (!e.target.closest(".filter-dropdown")) {
      dropdownPanels.forEach(function (panel) {
        panel.classList.remove("open");
      });
    }
  });

  dropdownPanels.forEach(function (panel) {
    panel.addEventListener("click", function (e) {
      e.stopPropagation();
    });
  });
});