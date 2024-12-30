let canvas = document.getElementById('canvas');
let ctx = canvas.getContext('2d');
let drawing = false;

// 캔버스 초기화
ctx.fillStyle = 'white';
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.strokeStyle = 'black';
ctx.lineWidth = 10;
ctx.lineCap = 'round';

// 마우스 이벤트 리스너
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

function startDrawing(e) {
    drawing = true;
    draw(e);
}

function draw(e) {
    if (!drawing) return;
    let rect = canvas.getBoundingClientRect();
    let x = e.clientX - rect.left;
    let y = e.clientY - rect.top;
    
    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
}

function stopDrawing() {
    drawing = false;
    ctx.beginPath();
}

function clearCanvas() {
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.beginPath();
    document.getElementById('prediction').textContent = '';
}

function predict() {
    // 캔버스 데이터를 이미지로 변환
    let imageData = canvas.toDataURL('image/png');
    
    // 서버로 예측 요청
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({image: imageData})
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('prediction').textContent = data.prediction;
    });
}
