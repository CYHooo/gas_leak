document.addEventListener("DOMContentLoaded", function () {
    // 获取SVG容器
    const svg = d3.select("#contourChart");
    const width = +svg.attr("width");
    const height = +svg.attr("height");

    // 定义网格大小
    const gridSize = 32;

    // 定义颜色比例尺
    const colorScale = d3.scaleSequential(d3.interpolateViridis)
        .domain([0, 1]); // 根据实际浓度范围调整

    // 定义等高线生成器
    const contours = d3.contours()
        .size([gridSize, gridSize])
        .thresholds(d3.range(0, 1, 0.01)); // 根据需要调整阈值

    // 定义比例转换
    const xScale = d3.scaleLinear()
        .domain([-100, 100])
        .range([0, width]);

    const yScale = d3.scaleLinear()
        .domain([-100, 100])
        .range([height, 0]); // 注意Y轴方向

    const path = d3.geoPath(d3.geoIdentity().scale(width / gridSize));

    // 动画函数
    function animate() {
        let timeIndex = 0;

        function drawFrame() {
            // 获取当前时间步的数据
            const data = concentrationData[timeIndex];
            const concentration = data.concentration.flat();

            // 生成等高线数据
            const contourData = contours(concentration);

            // 绑定数据并绘制路径
            const paths = svg.selectAll("path")
                .data(contourData);

            paths.enter().append("path")
                .merge(paths)
                .attr("d", path)
                .attr("fill", d => colorScale(d.value))
                .attr("stroke", "none");

            paths.exit().remove();

            // 更新时间步
            timeIndex = (timeIndex + 1) % concentrationData.length;

            // 每隔一定时间绘制下一帧
            setTimeout(drawFrame, 100); // 100毫秒，可根据需要调整
        }

        drawFrame();
    }

    // 开始动画
    animate();
});
