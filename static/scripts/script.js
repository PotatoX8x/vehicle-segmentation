img_input = document.getElementById("img_input")

img_input.addEventListener("change", (event) => {
    image = event.target.files[0]
    image_data = new FormData();
    image_data.append('image', image);
    
    fetch("/vehicle_segment", {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: image_data
    })
    .then(response => response.blob())
    .catch(error => {
        console.error('Error:', error);
    });
    
    img_output = document.getElementById("img_output")
    img_output.src = URL.createObjectURL(response);
})