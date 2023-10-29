/*!
 * Start Bootstrap - Grayscale v7.0.6 (https://startbootstrap.com/theme/grayscale)
 * Copyright 2013-2023 Start Bootstrap
 * Licensed under MIT (https://github.com/StartBootstrap/startbootstrap-grayscale/blob/master/LICENSE)
 */
//
// Scripts
//

window.addEventListener("DOMContentLoaded", (event) => {
  // Navbar shrink function
    var navbarShrink = function () {
        const navbarCollapsible = document.body.querySelector("#mainNav");
        if (!navbarCollapsible) {
            return;
        }
        if (window.scrollY === 0) {
            navbarCollapsible.classList.remove("navbar-shrink");
        } else {
            navbarCollapsible.classList.add("navbar-shrink");
        }
    };

    // Shrink the navbar
    navbarShrink();

    // Shrink the navbar when page is scrolled
    document.addEventListener("scroll", navbarShrink);

    // Activate Bootstrap scrollspy on the main nav element
    const mainNav = document.body.querySelector("#mainNav");
    if (mainNav) {
        new bootstrap.ScrollSpy(document.body, {
            target: "#mainNav",
            rootMargin: "0px 0px -40%",
        });
    }

    // Collapse responsive navbar when toggler is visible
    const navbarToggler = document.body.querySelector(".navbar-toggler");
    const responsiveNavItems = [].slice.call(
        document.querySelectorAll("#navbarResponsive .nav-link")
    );
    responsiveNavItems.map(function (responsiveNavItem) {
        responsiveNavItem.addEventListener("click", () => {
            if (window.getComputedStyle(navbarToggler).display !== "none") {
                navbarToggler.click();
            }
        });
    });
});
  
  
document.addEventListener("DOMContentLoaded", function () {
    var clickedSymptoms = {};

    var cards = document.querySelectorAll(".card");
    cards.forEach(function (card) {
        card.addEventListener("click", function () {
            card.classList.toggle("clicked");

            var symptom = card.getAttribute("data-symptom");

            clickedSymptoms[symptom] = !clickedSymptoms[symptom];

            updateHiddenInput();
        });
    });

    function updateHiddenInput() {
        var selectedSymptoms = Object.keys(clickedSymptoms).filter(function (symptom) {
            return clickedSymptoms[symptom];
        });

        document.getElementById("selectedSymptoms").value = selectedSymptoms.join(",");
    }
});

  // static/js/scripts.js
document.addEventListener("DOMContentLoaded", function () {
    var form = document.querySelector('form');
    form.addEventListener('submit', function (event) {
        event.preventDefault();

        // Serialize form data
        var formData = new FormData(form);

        fetch('/', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            showResultModal(data.result);

            clearSelectedSymptoms();
        })
        .catch(error => console.error('Error:', error));
    });

    function clearSelectedSymptoms() {
        var cards = document.querySelectorAll(".card");
        cards.forEach(function (card) {
            card.classList.remove("clicked");
        });

        document.getElementById("selectedSymptoms").value = "";
    }
});

function showResultModal(result) {
    document.getElementById('predictedDisease').innerText = 'Predicted Disease: ' + result;
    var resultModal = new bootstrap.Modal(document.getElementById('resultModal'));
    resultModal.show();
}