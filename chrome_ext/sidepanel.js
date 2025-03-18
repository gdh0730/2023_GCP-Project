document.getElementById("capture-btn").addEventListener("click", () => {   // 캡처, 다운로드 및 앱 엔진으로 전송
    chrome.runtime.sendMessage({ action: "captureAndDownload" });
  });


  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {

    if (message.action === "setImageUri") {
      const imageUrl = message.imageUrl;
  
      const imgElement = document.querySelector("#captured-image");
      imgElement.src = imageUrl;
      imgElement.style.display = "block";
      sendResponse({ status: "success" });
    }

    function initialize() {
      // container 요소 초기화
      const container = document.getElementById('container');
      container.innerHTML = '';
    }
  
  if(message.action == 'setServerResponse'){
    const serverResponse = message.serverResponse;

    // UI 엘리먼트 생성 및 추가
    const container = document.getElementById('container');
    const list = document.createElement('ul');

    // Flexbox 스타일 추가
    list.style.display = 'flex';
    list.style.flexWrap = 'wrap';
    list.style.gap = '10px';

    serverResponse.forEach(item => {
      //
      const listItem = document.createElement('li');
      listItem.style.flexBasis = '10%'; 
      
      //link 정보
      const link = document.createElement('a');
      link.href = item.link;

      // target="_blank" 속성 추가
      link.target="_blank";
      
      //썸네일 정보
      const thumbnail = document.createElement('img');
      thumbnail.src = item.thumbnail;
      
      //상품이름 정보
      const title = document.createElement('p');
      title.textContent = item.title;
      
      // 가격 정보
      const price = document.createElement('p');
      price.textContent = `가격: ${item.price}`;
      
      // UI 엘리먼트에 display 속성 설정
      listItem.style.display="block";
      link.style.display="block";
      title.style.display="block";
      price.style.display="block";
      
      link.appendChild(thumbnail);
      link.appendChild(title);
      link.appendChild(price);
      
      listItem.appendChild(link);
      list.appendChild(listItem);
    });

    container.appendChild(list);


  }});