const extensionAPI = typeof browser === 'undefined' ? chrome : browser;

chrome.extension.onRequest.addListener(function(prediction){
    if (prediction == 1){
        alert("Warning: Phishing detected!!");
    }
    else if (prediction == -1){
        // alert("No phishing detected");
        console.log("No phishing detected");
    }
});