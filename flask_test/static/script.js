document.addEventListener('DOMContentLoaded', () => {
    const fetchButton = document.getElementById('fetchButton');
    const sendButton = document.getElementById('sendButton');
    const responseContainer = document.getElementById('response');

    // Fetch Data Button
    fetchButton.addEventListener('click', async () => {
        try {
            const response = await fetch('/fetch-data', { method: 'GET' });
            const result = await response.json();
            responseContainer.textContent = JSON.stringify(result, null, 2);
        } catch (error) {
            responseContainer.textContent = `Error: ${error.message}`;
        }
    });

    // Send Data Button
    sendButton.addEventListener('click', async () => {
        try {
            const dataToSend = { title: "foo", body: "bar", userId: 1 };
            const response = await fetch('/send-data', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(dataToSend),
            });
            const result = await response.json();
            responseContainer.textContent = JSON.stringify(result, null, 2);
        } catch (error) {
            responseContainer.textContent = `Error: ${error.message}`;
        }
    });
});
