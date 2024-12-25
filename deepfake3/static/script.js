document.addEventListener('DOMContentLoaded', function() {
    const sourceInput = document.getElementById('source-image');
    const targetInput = document.getElementById('target-image');
    const sourcePreview = document.getElementById('source-preview');
    const targetPreview = document.getElementById('target-preview');
    const resultPreview = document.getElementById('result-preview');
    const blendButton = document.getElementById('blend-button');
    const errorMessage = document.getElementById('error-message');

    function previewImage(input, preview) {
        const file = input.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                const img = document.createElement('img');
                img.src = e.target.result;
                preview.innerHTML = '';
                preview.appendChild(img);
            }
            reader.readAsDataURL(file);
        }
    }

    sourceInput.addEventListener('change', function() {
        previewImage(this, sourcePreview);
    });

    targetInput.addEventListener('change', function() {
        previewImage(this, targetPreview);
    });

    blendButton.addEventListener('click', async function() {
        if (!sourceInput.files[0] || !targetInput.files[0]) {
            errorMessage.textContent = '두 이미지를 모두 선택해주세요.';
            return;
        }

        const formData = new FormData();
        formData.append('source', sourceInput.files[0]);
        formData.append('target', targetInput.files[0]);

        try {
            errorMessage.textContent = '';
            blendButton.disabled = true;
            blendButton.textContent = '처리중...';

            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                const resultImg = document.createElement('img');
                resultImg.src = `/static/uploads/${data.result}?t=${new Date().getTime()}`;
                resultPreview.innerHTML = '';
                resultPreview.appendChild(resultImg);
            } else {
                errorMessage.textContent = data.error;
            }
        } catch (error) {
            errorMessage.textContent = '처리 중 오류가 발생했습니다.';
        } finally {
            blendButton.disabled = false;
            blendButton.textContent = '블렌딩 시작';
        }
    });
});
