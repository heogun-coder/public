document.addEventListener("DOMContentLoaded", () => {
    console.log("DOM fully loaded and parsed");
    const form = document.getElementById("data-form");
    const responseDiv = document.getElementById("response");
    
    form.addEventListener("submit", async (event) => {
        event.preventDefault();
        console.log("Form submitted");

        // 폼 데이터 가져오기
        const title = document.getElementById("title").value;
        const body = document.getElementById("body").value;

        // 데이터 전송
        try {
            const response = await fetch("https://127.0.0.1:5000/send", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ title, body }),
            });

            const result = await response.json();

            // 응답 표시
            if (response.ok) {
                responseDiv.innerHTML = `
                    <p class="success">${result.message}</p>
                    <p><strong>Data Sent:</strong> ${JSON.stringify(result.data_sent)}</p>
                    <p><strong>Server Response:</strong> ${JSON.stringify(result.server_response)}</p>
                `;
            } else {
                responseDiv.innerHTML = `
                    <p class="error">Error: ${result.error}</p>
                    <p><strong>Details:</strong> ${JSON.stringify(result.details || "Unknown error")}</p>
                `;
            }
        } catch (error) {
            responseDiv.innerHTML = `<p class="error">Error: ${error.message}</p>`;
            console.error("Error during fetch:", error);
        }
    });
});
