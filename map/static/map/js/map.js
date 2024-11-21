src="https://maps.googleapis.com/maps/api/js?key=AIzaSyB1jrXysy_wzdu6jjk7JJoAmaZpROtSeaA&callback=initMap&v=weekly"

// Initialize and add the map
let map;

function initMap() {
    // The location of sk energy
    const position = { lat: 35.488079, lng: 129.361921 };
  

    // The map, centered at sk energy
    map = new google.maps.Map(document.getElementById("map"), {
        zoom: 18,
        center: position,
        disableDefaultUI: false,
        mapTypeId:'satellite',
        mapId: "DEMO_MAP_ID",
    });

}

window.initMap = initMap;


