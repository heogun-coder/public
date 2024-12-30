document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('uploadForm');
    const alphaSlider = document.getElementById('alpha');
    const alphaValue = document.getElementById('alphaValue');
    const resultImage = document.getElementById('resultImage');
    const blendButton = document.querySelector('.blend-button');
    
    function showLoading() {
        blendButton.disabled = true;
        blendButton.textContent = '처리중...';
        resultImage.innerHTML = '<div class="loading"></div>';
    }
    
    function hideLoading() {
        blendButton.disabled = false;
        blendButton.textContent = '이미지 블렌드';
    }
    
    function showError(message) {
        resultImage.innerHTML = `<div class="error">${message}</div>`;
    }
    
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
        const image1 = document.getElementById('image1').files[0];
        const image2 = document.getElementById('image2').files[0];
        
        if (!image1 || !image2) {
            showError('두 개의 이미지를 모두 선택해주세요.');
            return;
        }
        
        formData.append('image1', image1);
        formData.append('image2', image2);
        formData.append('alpha', alphaSlider.value);
        
        showLoading();
        
        try {
            const response = await fetch('/blend', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || '이미지 처리 중 오류가 발생했습니다.');
            }
            
            const blob = await response.blob();
            const imageUrl = URL.createObjectURL(blob);
            resultImage.innerHTML = `<img src="${imageUrl}" alt="Blended Image">`;
            
        } catch (error) {
            console.error('Error:', error);
            showError(error.message);
        } finally {
            hideLoading();
        }
    });
});