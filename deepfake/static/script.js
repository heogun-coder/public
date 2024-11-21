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

function validateYouTubeUrl(url) {
    const pattern = /^(https?:\/\/)?(www\.)?(youtube\.com\/watch\?v=|youtu\.be\/)[a-zA-Z0-9_-]{11}$/;
    return pattern.test(url);
}

function validateYouTubeUrl(url) {
    // YouTube URL 패턴 확장
    const patterns = [
        /^(https?:\/\/)?(www\.)?(youtube\.com\/watch\?v=|youtu\.be\/)[a-zA-Z0-9_-]{11}/, // 기본 URL
        /^(https?:\/\/)?(www\.)?youtube\.com\/shorts\/[a-zA-Z0-9_-]{11}/, // Shorts URL
        /^(https?:\/\/)?(www\.)?youtube\.com\/embed\/[a-zA-Z0-9_-]{11}/, // Embed URL
    ];
    
    return patterns.some(pattern => pattern.test(url));
} 

function processVideos() {
    const sourceType = document.getElementById('sourceType').value;
    const targetType = document.getElementById('targetType').value;
    
    const formData = new FormData();
    formData.append('source_type', sourceType);
    formData.append('target_type', targetType);

    // 소스 비디오 처리
    if (sourceType === 'file') {
        const sourceFile = document.getElementById('sourceFile').files[0];
        if (!sourceFile) {
            alert('원본 영상 파일을 선택해주세요.');
            return;
        }
        formData.append('source_file', sourceFile);
    } else {
        const sourceUrl = document.getElementById('sourceUrl').value.trim();
        if (!validateYouTubeUrl(sourceUrl)) {
            alert('유효한 YouTube URL을 입력해주세요.');
            return;
        }
        formData.append('source_url', sourceUrl);
    }

    // 타겟 비디오 처리
    if (targetType === 'file') {
        const targetFile = document.getElementById('targetFile').files[0];
        if (!targetFile) {
            alert('타겟 영상 파일을 선택해주세요.');
            return;
        }
        formData.append('target_file', targetFile);
    } else {
        const targetUrl = document.getElementById('targetUrl').value.trim();
        if (!validateYouTubeUrl(targetUrl)) {
            alert('유효한 YouTube URL을 입력해주세요.');
            return;
        }
        formData.append('target_url', targetUrl);
    }

    // UI 초기화
    document.getElementById('progressContainer').style.display = 'block';
    document.getElementById('progressBar').style.width = '0%';
    document.getElementById('progressMessage').textContent = '처리 시작...';
    document.getElementById('progressPercent').textContent = '0%';
    document.getElementById('resultVideo').style.display = 'none';

    // 서버로 전송
    fetch('/process_videos', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(data => {
                throw new Error(data.error || '처리 중 오류가 발생했습니다.');
            });
        }
        return response.blob();
    })
    .then(blob => {
        const videoUrl = URL.createObjectURL(blob);
        const resultVideo = document.getElementById('resultVideo');
        resultVideo.src = videoUrl;
        resultVideo.style.display = 'block';
        document.getElementById('progressContainer').style.display = 'none';
    })
    .catch(error => {
        alert(error.message);
        document.getElementById('progressContainer').style.display = 'none';
    });
}

// 소켓 이벤트 리스너
socket.on('progress_update', function(data) {
    document.getElementById('progressBar').style.width = data.progress + '%';
    document.getElementById('progressMessage').textContent = data.message;
    document.getElementById('progressPercent').textContent = data.progress + '%';
});