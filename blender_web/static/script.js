document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('uploadForm');
    const alphaSlider = document.getElementById('alpha');
    const alphaValue = document.getElementById('alphaValue');
    
    // 이미지 미리보기 함수
    function previewImage(input, previewId) {
        const preview = document.getElementById(previewId);
        const file = input.files[0];
        
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                preview.innerHTML = `<img src="${e.target.result}" alt="Preview">`;
            }
            reader.readAsDataURL(file);
        }
    }
    
    // 이미지 입력 이벤트 리스너
    document.getElementById('image1').addEventListener('change', function() {
        previewImage(this, 'preview1');
    });
    
    document.getElementById('image2').addEventListener('change', function() {
        previewImage(this, 'preview2');
    });
    
    // 알파값 슬라이더 이벤트
    alphaSlider.addEventListener('input', function() {
        alphaValue.textContent = this.value;
    });
    
    // 폼 제출 처리
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const formData = new FormData();
        formData.append('image1', document.getElementById('image1').files[0]);
        formData.append('image2', document.getElementById('image2').files[0]);
        formData.append('alpha', alphaSlider.value);
        
        try {
            const response = await fetch('/blend', {
                method: 'POST',
                body: formData
            });
            
            if (response.ok) {
                const blob = await response.blob();
                const imageUrl = URL.createObjectURL(blob);
                document.getElementById('resultImage').innerHTML = 
                    `<img src="${imageUrl}" alt="Blended Image">`;
            } else {
                alert('이미지 블렌딩 중 오류가 발생했습니다.');
            }
        } catch (error) {
            console.error('Error:', error);
            alert('서버 통신 중 오류가 발생했습니다.');
        }
    });
});