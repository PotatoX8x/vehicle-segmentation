const imgInput = document.getElementById("img_input");

imgInput.addEventListener("change", (event) => {
    const image = event.target.files[0];
    const formData = new FormData();
    formData.append('file', image);

    fetch("/vehicle_segment", {
        method: 'POST',
        body: formData
    })
    .then(response => response.blob())
    .then(blob => {
        const imgOutput = document.getElementById("img_output");
        imgOutput.src = URL.createObjectURL(blob);
    })
    .catch(error => {
        console.error('Error:', error);
    });
});
