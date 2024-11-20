const socket = io();

function toggleSourceInput() {
    const sourceType = document.getElementById('sourceType').value;
    document.getElementById('sourceFileInput').style.display = 
        sourceType === 'file' ? 'block' : 'none';
    document.getElementById('sourceUrlInput').style.display = 
        sourceType === 'youtube' ? 'block' : 'none';
}

function toggleTargetInput() {
    const targetType = document.getElementById('targetType').value;
    document.getElementById('targetFileInput').style.display = 
        targetType === 'file' ? 'block' : 'none';
    document.getElementById('targetUrlInput').style.display = 
        targetType === 'youtube' ? 'block' : 'none';
}

function processVideos() {
    const formData = new FormData();
    
    const sourceType = document.getElementById('sourceType').value;
    const targetType = document.getElementById('targetType').value;
    
    formData.append('source_type', sourceType);
    formData.append('target_type', targetType);
    
    if (sourceType === 'file') {
        const sourceFile = document.getElementById('sourceFile').files[0];
        if (!sourceFile) {
            alert('원본 영상을 선택해주세요.');
            return;
        }
        formData.append('source_file', sourceFile);
    } else {
        const sourceUrl = document.getElementById('sourceUrl').value;
        if (!sourceUrl) {
            alert('원본 영상 URL을 입력해주세요.');
            return;
        }
        formData.append('source_url', sourceUrl);
    }
    
    if (targetType === 'file') {
        const targetFile = document.getElementById('targetFile').files[0];
        if (!targetFile) {
            alert('타겟 영상을 선택해주세요.');
            return;
        }
        formData.append('target_file', targetFile);
    } else {
        const targetUrl = document.getElementById('targetUrl').value;
        if (!targetUrl) {
            alert('타겟 영상 URL을 입력해주세요.');
            return;
        }
        formData.append('target_url', targetUrl);
    }
    
    document.getElementById('progressContainer').style.display = 'block';
    document.getElementById('progressBar').style.width = '0%';
    document.getElementById('progressMessage').textContent = '처리 시작 중...';
    
    fetch('/process_videos', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(error => {
                throw new Error(error.error || '처리 중 오류가 발생했습니다.');
            });
        }
        return response.blob();
    })
    .then(blob => {
        const videoUrl = URL.createObjectURL(blob);
        const resultVideo = document.getElementById('resultVideo');
        resultVideo.src = videoUrl;
        resultVideo.style.display = 'block';
        resultVideo.play();
    })
    .catch(error => {
        alert(error.message);
    })
    .finally(() => {
        document.getElementById('progressContainer').style.display = 'none';
    });
}

socket.on('progress_update', function(data) {
    document.getElementById('progressBar').style.width = data.progress + '%';
    document.getElementById('progressMessage').textContent = data.message;
});

socket.on('error', function(data) {
    alert('에러: ' + data.message);
    document.getElementById('progressContainer').style.display = 'none';
});