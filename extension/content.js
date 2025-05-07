// const extensionAPI = typeof browser === 'undefined' ? chrome : browser;
let checked = [];
let currentDomain = document.domain;

function scanTextContent() {
  console.log("Scanning text on:", currentDomain);

  let elements = segments(document.body);
  for (let i = 0; i < elements.length; i++) {
    if (!(elements[i] && elements[i].innerText)) continue;
    if (checked.includes(elements[i])) continue;

    const text = elements[i].innerText.trim();
    if (text.length === 0) continue;

    checkTextForHate(text, elements[i]);
    checked.push(elements[i]);
  }
}

async function checkTextForHate(text, element) {
  try {
    const response = await fetch('http://127.0.0.1:5000/predict-text', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      mode: 'cors',
      body: JSON.stringify({ text }),
    });

    const data = await response.json();
    console.log("Text API:", data);

    if ((data.class_index === 0 || data.class_index === 1) && data.confidence > 0.5) {
      element.style.backgroundColor = "#ffcccc";
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

    const response = await fetch('http://127.0.0.1:5000/predict-image', {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();
    // console.log("Image API:", data);

    // index 1: 
    // 2: neutral
    // 4: sexy

    if (data.class_index === 1 || data.class_index === 4) {
      console.log("NSFW detected:", imageUrl);
      console.log("NSFW confidence:", data.confidence);
      imgElement.style.filter = "blur(5px)";
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


// window.addEventListener("load", () => {
//   scanTextContent();
//   scanImages();
// });

window.addEventListener('load', async function () {



  const images = document.querySelectorAll("img");
  const imageUrls = Array.from(images).map(img => ({ url: img.src, element: img }));

  const tasks = imageUrls.map(({ url, element }) => checkImageForNSFW(url, element));
  await Promise.allSettled(tasks);

  scanTextContent();

  testdata = [isIPInURL(),isLongURL(),isTinyURL(),isAlphaNumericURL(),isRedirectingURL(),isHypenURL(),isMultiDomainURL(),isFaviconDomainUnidentical(),isIllegalHttpsURL(),isImgFromDifferentDomain(),isAnchorFromDifferentDomain(),isScLnkFromDifferentDomain(),isFormActionInvalid(),isMailToAvailable(),isStatusBarTampered(),isIframePresent()];

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