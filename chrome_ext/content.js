// 이미지 캡쳐 함수
function captureImage() {
  const video = document.querySelector("video");
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg");
}

  // 이미지 다운로드 함수
function downloadImage(dataUri, fileName) {
  const link = document.createElement("a");
  link.href = dataUri;
  link.download = fileName;
  link.target = "_blank";
  link.click();
}

// 이미지 데이터 전송 함수
const serverUrl="https://jnu-idv-03.du.r.appspot.com/detect_clothing" // image=@jpg

async function sendImageToServer(blob) { 
  const formData = new FormData();
  formData.append("image", blob, "image.jpeg");

  try {
    const response = await fetch(serverUrl, {
      method: "POST",
      body: formData,
      mode :'cors'
    });

    if (response.ok) {
      console.log("이미지 전송 성공");
      const jsonResponse = await response.json(); // JSON 형식으로 변경
      const results = jsonResponse.results; // "results" 항목 추출
      console.log(`서버 응답 내용: ${results}`);
      
      let result_2return = [];
      results.forEach((item) => {
                let thumbnailURL = item.thumbnail;
                result_2return.push({title: item.title, link: item.link, price: item.price.value, thumbnail: thumbnailURL})
      });
      console.log(result_2return)
      
    // response result 표시
    chrome.runtime.sendMessage({
      action: "setServerResponse",
      serverResponse: result_2return, // 서버 응답 결과를 전달합니다
    });

      
    } else {
      console.error(`이미지 전송 실패: ${response.status} ${response.statusText}`);
    }
  } catch (error) {
    console.error(`이미지에서 옷을 찾을수없습니다.: ${error}`);
  }
}


function dataURItoBlob(dataURI) { //Base64 인코딩된 이미지 데이터를 Blob 객체로 변환하여 반환
  const byteString = atob(dataURI.split(",")[1]);
  const mimeString = dataURI.split(",")[0].split(":")[1].split(";")[0];
  const arrayBuffer = new ArrayBuffer(byteString.length);
  const uintArray = new Uint8Array(arrayBuffer);
  for (let i = 0; i < byteString.length; i++) {
    uintArray[i] = byteString.charCodeAt(i);
  }
  return new Blob([arrayBuffer], { type: mimeString });
}

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "captureAndDownload") {
    try {
      const dataUri = captureImage();
      downloadImage(dataUri, "screenshot.jpeg"); // 이미지 다운로드
      const blob = dataURItoBlob(dataUri);
      sendImageToServer(blob);

    // 캡쳐된 이미지 표시
    chrome.runtime.sendMessage({
      action: "setImageUri",
      imageUrl: dataUri,
    });   
      

    sendResponse({ success: true });
    } catch (error) {
      console.error("Failed to capture image", error);
      sendResponse({ success: false });
    }
  }
});