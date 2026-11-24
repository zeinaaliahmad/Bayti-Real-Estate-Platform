document.addEventListener("DOMContentLoaded", function () {
  console.log("carousel.js loaded");

  let currentSlide = 0;
  const slides = document.querySelectorAll(".carousel-slide");

  console.log("slides found:", slides.length);

  function showSlide(index) {
    slides.forEach((slide) => {
      slide.classList.remove("active");
    });
    slides[index].classList.add("active");
    console.log("showing slide:", index);
  }

  function nextSlide() {
    currentSlide = (currentSlide + 1) % slides.length;
    showSlide(currentSlide);
  }

  if (slides.length > 0) {
    showSlide(currentSlide);
    setInterval(nextSlide, 4000);
  }
});