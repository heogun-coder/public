document.addEventListener('DOMContentLoaded', () => {
    const button = document.getElementById('sendRequest');

    button.addEventListener('click', async () => {
        const data = { message: "Hello from the client!" };

        try {
            const response = await fetch('/api/data', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data),
            });

            const result = await response.json();
            console.log("Server Response:", result);
        } catch (error) {
            console.error("Error sending request:", error);
        }
    });
});
