async function listModels() {
    const apiKey = "AIzaSyCqHK8Y2953PWZSbnno-TdqGcT01X9Mzt0";
    try {
        const request = await fetch(`https://generativelanguage.googleapis.com/v1beta/models?key=${apiKey}`);
        const data = await request.json();
        if (data.models) {
            console.log("Available models:");
            data.models.forEach(m => console.log(`- ${m.name} (${m.displayName})`));
        } else {
            console.log("Error or no models found:");
            console.log(JSON.stringify(data, null, 2));
        }
    } catch (e) {
        console.error(e);
    }
}

listModels();
