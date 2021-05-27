$(document).ready(function(){
    var images = ['homeimg.jpg','homeimg.jpeg','homeimg.webp']
    for (var img of images) {
        $("#card_container").append(`<div class="card col">
        <img  src="static/`+img+`" class="card-img-top hal" alt="...">
      </div>`
        )
    }

})



function openCamera() {
    $.ajax({
        url: '/opencamera',

        success: function (data) {
            $("#emotion_result").append("<h1>" + data + "</h1>")
        },


        error: function () {
        },
    });
}

