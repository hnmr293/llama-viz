(function() {

    const info = document.querySelector(".output_token_info");
    
    const ele = document.querySelector("#output");
    ele.addEventListener("mouseover", e => {
        let target = e.target;
        if (target.classList.contains("special")) target = target.parentNode;
        if (!target.classList.contains("token")) return;
        const d = target.dataset;
        info.innerHTML = `index ${d.tokenPos}<br/>id ${d.tokenId}`;
    });
    ele.addEventListener("mouseout", e => {
        if (!e.target.classList.contains("token")) return;
        info.innerHTML = "";
    });

    document.addEventListener("keydown", e => {
        if (e.ctrlKey && (e.code == "Enter" || e.code == "NumpadEnter")) {
            const button = document.querySelector("button.primary");
            if (button) button.click();
        }
    });

})
