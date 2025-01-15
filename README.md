# Rendervous - Rendering and Learning and vice versa

Rendervous is a project designed primarily for academic purposes. The core idea is the integration of GPU rendering capabilities including ray-tracing (Vulkan) with deep learning (Pytorch).

As a result you have a differentiable renderer that can be included in learning models and vice versa, learning models that can be included as renderer components (materials, scattering functions, parameters, etc).




## Dependencies

- torch
- cffi
- pywin32 (Windows)
- cuda-python (if cuda device can be used)

### Secondary dependencies
- matplotlib (most of the offline-rendering examples)

### Interactive examples 
- imgui
- glfw (for interactive examples)


<table>
<tr> 
<td>
    <b>Introducing rendervous:</b> Creating maps in rendervous. Manipulating vectors and operations.<br/>
    <a href="https://colab.research.google.com/github/rendervous/rendervous_project/blob/main/tutorials/e01_introducing_rendervous.ipynb">open in colab</a>
</td>
</tr>

<tr> 
<td>
    <b>Differentiability:</b> Testing differentiability with simple regressions.<br/>
    <a href="https://colab.research.google.com/github/rendervous/rendervous_project/blob/main/tutorials/e02_differentiability.ipynb">open in colab</a>
</td>
</tr>

<tr> 
<td>
    <b>Simple MLP:</b> Chaining maps to build an MLP.<br/>
    <a href="https://colab.research.google.com/github/rendervous/rendervous_project/blob/main/tutorials/e03_basic_mlp.ipynb">open in colab</a>
</td>
</tr>

<tr> 
<td>
    <b>Sensors:</b> Examples of sensors.<br/>
    <a href="https://colab.research.google.com/github/rendervous/rendervous_project/blob/main/tutorials/e04_sensors.ipynb">open in colab</a>
</td>
</tr>

<tr> 
<td>
    <b>Transmittances:</b> Example of transmittance computation through a volume.<br/>
    <a href="https://colab.research.google.com/github/rendervous/rendervous_project/blob/main/tutorials/e05_transmittances.ipynb">open in colab</a>
</td>
</tr>

<tr> 
<td>
    <b>Reconstruction:</b> Example of volume reconstruction from transmittance.<br/>
    <a href="https://colab.research.google.com/github/rendervous/rendervous_project/blob/main/tutorials/e06_reconstruction_from_transmittance.ipynb">open in colab</a>
</td>
</tr>

</table> 