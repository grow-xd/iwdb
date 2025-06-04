// const extensionAPI = typeof browser === 'undefined' ? chrome : browser;
let currentDomain = document.domain;
let checked = new WeakSet();

function scanTextContent() {
  console.log("Scanning text on:", currentDomain);
  let elements = segments(document.body);
  for (let i = 0; i < elements.length; i++) {
    if (!(elements[i] && elements[i].innerText)) continue;
    if (checked.has(elements[i])) continue;
    const text = elements[i].innerText.trim();
    if (text.length === 0) continue;
    checkTextForHate(text, elements[i]);
    checked.add(elements[i]);
  }
}

async function checkTextForHate(text, element) {
  try {
    const response = await fetch('http://127.0.0.1:5000/predict-text', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      mode: 'cors',
      body: JSON.stringify({ text, domain: currentDomain }),
    });

    const data = await response.json();
    console.log("Text API:", data);

    if ((data.class_index === 0 || data.class_index === 1) && data.confidence > 0.5) {
      element.style.backgroundColor = window.getComputedStyle(element).color;
      element.title = `⚠ ${data.predicted_label} (${Math.round(data.confidence * 100)}%)`;
    }
  } catch (error) {
    console.error("Text check error:", error);
  }
}

function scanImages() {
  const images = document.querySelectorAll("img");
  images.forEach((img, index) => {
    if (checked.includes(img)) return;
    checked.push(img);

    setTimeout(() => {
      checkImageForNSFW(img.src, img);
    }, index * 2000); // rate-limit
  });
}

async function checkImageForNSFW(imageUrl, imgElement) {
  try {
    const formData = new FormData();
    formData.append("image_url", imageUrl);
    formData.append("domain", currentDomain);

    const response = await fetch('http://127.0.0.1:5000/predict-image', {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();

    if (data.predicted_class == "sexy" || data.predicted_class == "hentai" || data.predicted_class == "porn") {
      console.log("NSFW detected:", imageUrl);
      console.log("NSFW confidence:", data.confidence);
      imgElement.style.filter = "blur(10px)";
      imgElement.style.border = "5px solid red";
      const labelDiv = document.createElement("div");
      labelDiv.textContent = `⚠ NSFW (${Math.round(data.confidence * 100)}%)`;
      labelDiv.style.color = "red";
      labelDiv.style.fontWeight = "bold";
      imgElement.parentNode.insertBefore(labelDiv, imgElement);
    }
  } catch (error) {
    console.error("Image check error:", error);
  }
}


function blurVideo(video) {
  if (!video.classList.contains("blurred")) {
    video.style.filter = "blur(20px)";
    video.classList.add("blurred");
  }
}

function monitorVideo(video) {
  if (checked.has(video)) return;
  checked.add(video);

  video.crossOrigin = "anonymous";
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d", { willReadFrequently: true });

  const interval = setInterval(() => {
    try {
      if (video.readyState >= 2 && video.videoWidth && video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        canvas.toBlob(async (blob) => {
          if (!blob) return;
          const formData = new FormData();
          formData.append("image", blob, "frame.jpg");
          formData.append("domain", currentDomain);
          try {
            const res = await fetch("http://127.0.0.1:5000/predict-image", {
              method: "POST",
              body: formData,
            });
            const data = await res.json();
            if (["sexy", "hentai", "porn"].includes(data.predicted_class)) {
              blurVideo(video);
              clearInterval(interval);
            }
          } catch (e) {
            console.error("Video frame API error:", e);
          }
        }, "image/jpeg", 0.7);
      }
    } catch (e) {
      console.warn("Canvas draw error:", e.message);
    }
  }, 3000); // every 1s
}

function scanImages() {
  const images = document.querySelectorAll("img");
  images.forEach((img, index) => {
    if (checked.has(img)) return;
    checked.add(img);
    setTimeout(() => checkImageForNSFW(img.src, img), index * 2000);
  });
}

function scanVideos() {
  const videos = document.querySelectorAll("video");
  videos.forEach(monitorVideo);
}

function watchNewVideos() {
  const observer = new MutationObserver(() => scanVideos());
  observer.observe(document.body, {
    childList: true,
    subtree: true,
  });
}

window.addEventListener('load', async function () {

  scanImages();
  scanTextContent();
  scanVideos();
  watchNewVideos();

  testdata = [isIPInURL(), isLongURL(), isTinyURL(), isAlphaNumericURL(), isRedirectingURL(), isHypenURL(), isMultiDomainURL(), isFaviconDomainUnidentical(), isIllegalHttpsURL(), isImgFromDifferentDomain(), isAnchorFromDifferentDomain(), isScLnkFromDifferentDomain(), isFormActionInvalid(), isMailToAvailable(), isStatusBarTampered(), isIframePresent()];
  prediction = predict(testdata);
  chrome.extension.sendRequest(prediction);

});



// function handleChanges() {
//   scantexts();
// }
// const observer = new MutationObserver(handleChanges);

// const config = { childList: true, subtree: true };

// observer.observe(document.body, config);

// handleChanges();