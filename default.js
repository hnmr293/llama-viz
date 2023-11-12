(function() {

    const info = document.querySelector(".output_token_info");
    
    const eles = document.querySelectorAll(".output");
    for (let ele of eles) {
        ele.addEventListener("mouseover", e => {
            let target = e.target;
            if (target.classList.contains("special")) target = target.parentNode;
            if (!target.classList.contains("token")) return;
            if (info) {
                const d = target.dataset;
                info.innerHTML = `index ${d.tokenPos}<br/>id ${d.tokenId}`;
            }
        });
        ele.addEventListener("mouseout", e => {
            if (!e.target.classList.contains("token")) return;
            if (info) {
                info.innerHTML = "";
            }
        });
    }

    document.addEventListener("keydown", e => {
        if (e.ctrlKey && (e.code == "Enter" || e.code == "NumpadEnter")) {
            const button = document.querySelector("button.primary#run");
            if (button) {
                e.preventDefault();
                e.stopPropagation();
                button.click();
            }
        }
    }, { capture: true });

    document.querySelector(".output.attn").addEventListener("mousedown", e => {
        let target = e.target;
        if (target.classList.contains("special")) target = target.parentNode;
        if (!target.classList.contains("token")) return;
        target.classList.toggle("selected");
});

})
