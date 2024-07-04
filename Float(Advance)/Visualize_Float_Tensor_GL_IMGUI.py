import torch
import glfw
from OpenGL.GL import *
import numpy as np
from cuda import cudart as cu
import time
import imgui
from imgui.integrations.glfw import GlfwRenderer
import math



class GUI:
    def __init__(self, tensor, width=1000, heigh=600):
        self.tensor = tensor
        self.elementSize = tensor.element_size()
        self.tensorHeight,self.tensorWidth = tensor.shape
        self.windowWidth = width
        self.windowHeigh = heigh
        self.selectedX = 0
        self.selectedY = 0
        self.inputValue = 0.0
        self.applyToAll = False
        ## Vertex shader source code
        self.VERTEX_SHADER = """
        #version 330 core

        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec2 aTexCoord;

        uniform float uSize;
        uniform float uSizeData;
        uniform float uLocx;
        uniform float uLocy;
        
        out vec2 TexCoord;
        out float Isdata;
        void main()
        {
            TexCoord = aTexCoord;
            if (aPos.z == -1.0){
                gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);
                gl_PointSize = uSizeData;
                TexCoord = vec2(uLocx,uLocy);
                Isdata = 1.0;

            }
            else{
                gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);
                gl_PointSize = uSize;
                Isdata = 0.0;
            }
        }
        """
        ## Fragment shader source code
        self.FRAGMENT_SHADER = """
        #version 330 core

        out vec4 FragColor;

        in vec2 TexCoord;
        in float Isdata;

        uniform sampler2D ourTexture;

        void main()
        {
            FragColor = vec4(texture(ourTexture, TexCoord).rrr, 1.0f);
        }
        """

    def imguiUI(self):
        if imgui.is_mouse_down():
            first_mouse_x, first_mouse_y = imgui.get_mouse_pos()
            if (first_mouse_x >= self.windowWidth/8 and first_mouse_x <= 7*self.windowWidth/8) and (first_mouse_y >= 0 and first_mouse_y <= self.windowHeigh):
                mouse_x = first_mouse_x
                mouse_y = first_mouse_y
                self.selectedX = math.ceil(math.floor((mouse_x - 1/8*self.windowWidth) / ((6/8)*self.windowWidth/(2*(self.tensorWidth+1)))) / 2)-1
                if self.selectedX <= -1: self.selectedX = 0
                if self.selectedX >= self.tensorWidth: self.selectedX = self.tensorWidth-1
                self.selectedY =  self.tensorHeight - math.ceil(math.floor(mouse_y/(self.windowHeigh/(2*(self.tensorHeight+1)))) / 2)
                if self.selectedY <= -1: self.selectedY = 0
                if self.selectedY >= self.tensorHeight: self.selectedY = self.tensorHeight-1

        glUniform1f(self.uniform_location_locx, 1/(self.tensorWidth+1)*(self.selectedX+1))
        glUniform1f(self.uniform_location_locy, 1/(self.tensorHeight+1)*(self.selectedY+1))
        imgui.set_next_window_position(0, 0)
        imgui.set_next_window_size(self.windowWidth/8, self.windowHeigh)
        # window_bg_color = imgui.get_style().colors[imgui.COLOR_WINDOW_BACKGROUND]
        # imgui.get_style().colors[imgui.COLOR_WINDOW_BACKGROUND] = (*window_bg_color[:3], 1.0)
        imgui.get_style().colors[imgui.COLOR_WINDOW_BACKGROUND] = (0.18,0.18,0.18, 1.0)
        style = imgui.get_style()
        flags = imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_RESIZE
        with imgui.begin("Vertecies:",flags=flags):
                imgui.text("N:"+str(self.tensorWidth*self.tensorHeight))
                _, self.selectedY = imgui.input_int(':row', self.selectedY)
                _, self.selectedX = imgui.input_int(':col', self.selectedX)
                if self.selectedY <= -1:self.selectedY = self.tensorHeight- 1
                if self.selectedY >= self.tensorHeight:self.selectedY = 0 
                if self.selectedX <= -1:self.selectedX = self.tensorWidth-1 
                if self.selectedX >= self.tensorWidth:self.selectedX = 0 
                if self.tensorWidth*self.tensorHeight<=20000:
                    for i in range(self.tensorHeight):
                        for j in range(self.tensorWidth):
                            if i == self.selectedY and j==self.selectedX:
                                style.colors[imgui.COLOR_BUTTON] = (0.03, 0.07, 0.22, 1.0)
                                style.colors[imgui.COLOR_TEXT] = (1.0, 0.0, 0.0, 1.0)
                            else:
                                style.colors[imgui.COLOR_BUTTON] = (0.13, 0.27, 0.42, 1.0)
                                style.colors[imgui.COLOR_TEXT] = (1.0, 1.0, 1.0, 1.0)
                            if (imgui.button("vertex_"+str(i)+"_"+str(j),100,25)):
                                self.selectedX = j
                                self.selectedY = i
                else:
                    imgui.text("Generate over \n 20000 button \n in python take \n alot of time\n and drop fps")
        style.colors[imgui.COLOR_BUTTON] = (0.13, 0.27, 0.42, 1.0)
        style.colors[imgui.COLOR_TEXT] = (1.0, 1.0, 1.0, 1.0)
        imgui.set_next_window_position(self.windowWidth/8, 0)
        imgui.set_next_window_size(self.windowWidth-2*self.windowWidth/8, self.windowHeigh)
        imgui.get_style().colors[imgui.COLOR_WINDOW_BACKGROUND] = (0.18,0.18,0.18, 0.01)
        flags = imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_RESIZE
        with imgui.begin("output",flags=flags):
            draw_list = imgui.get_window_draw_list()
            thicknes = 5
            if self.tensorHeight*self.tensorWidth<=100:
                size = 30
            elif self.tensorHeight*self.tensorWidth<=200:
                size = 25
            elif self.tensorHeight*self.tensorWidth<=10000:
                thicknes = 2
                size = 5
            else:
                thicknes = 2
                size = 4
            posCircleX = 1/8 * self.windowWidth + (6/8) * self.windowWidth / (self.tensorWidth+1) * (self.selectedX+1)
            posCircleY = self.windowHeigh/(self.tensorHeight+1) * (self.tensorHeight-self.selectedY)
            draw_list.add_circle(posCircleX, posCircleY, size, imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 10.0),100, thicknes)
            ## invisible button
            # style.colors[imgui.COLOR_BUTTON] = (0.03, 0.07, 0.22, 0.0)
            # style.colors[imgui.COLOR_BUTTON_ACTIVE] = (0.03, 0.07, 0.22, 0.0)
            # style.colors[imgui.COLOR_BUTTON_HOVERED] = (0.03, 0.07, 0.22, 0.0)
            # if (imgui.button("",width_window-2*width_window/8,(97/100)*self.windowHeigh)):
        imgui.set_next_window_position(7*self.windowWidth/8, 2*self.windowHeigh/3)
        imgui.set_next_window_size(self.windowWidth/8, self.windowHeigh/3)
        imgui.get_style().colors[imgui.COLOR_WINDOW_BACKGROUND] = (0.18,0.18,0.18, 1.0)
        flags = imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_RESIZE
        with imgui.begin("Change:",flags=flags):
            imgui.text("New value:")
            _, self.inputValue = imgui.input_float('', self.inputValue)
            if (imgui.button("Change",100,25)):
                self.tensor[self.selectedY][self.selectedX]=self.inputValue
            _,self.applyToAll = imgui.checkbox("Change value \n if has been \n selected", self.applyToAll)
            if self.applyToAll:
                self.tensor[self.selectedY][self.selectedX]=self.inputValue 
        imgui.set_next_window_position(7*self.windowWidth/8, 0)
        imgui.set_next_window_size(self.windowWidth/8, self.windowHeigh/3)
        imgui.get_style().colors[imgui.COLOR_WINDOW_BACKGROUND] = (0.18,0.18,0.18, 1.0)
        flags = imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_RESIZE
        if self.selectedY != -1 and self.selectedX != -1:
            with imgui.begin("Information:",flags=flags):
                imgui.text("Value:")
                imgui.text(str(self.tensor[self.selectedY][self.selectedX].item()))

    def renderOpenGL(self):
        vertices = np.zeros((self.tensorHeight*self.tensorWidth*5+5),dtype=np.float32)
        index=0
        for i in range(self.tensorWidth):
            for j in range(self.tensorHeight):
                vertices[index+0] = -1+1/4+(i+1)*2*(6/8)/(self.tensorWidth+1)
                vertices[index+1] = -1+(j+1)*2/(self.tensorHeight+1)
                vertices[index+2] = 0.0
                vertices[index+3] = (i+1)/(self.tensorWidth+1)
                vertices[index+4] = (j+1)/(self.tensorHeight+1)
                index +=5
        vertices[index+0] = 7/8
        vertices[index+1] = 0.0
        vertices[index+2] = -1.0
        vertices[index+3] = -1+(0+1)/(self.tensorWidth+1)
        vertices[index+4] = -1+(0+1)/(self.tensorHeight+1)
        ##  Callback function for window resize
        def framebuffer_size_callback(window, width, height):
            glViewport(0, 0, width, height)
            self.windowWidth = width
            self.windowHeigh = height
            glUniform1f(self.uniform_location_size_data, self.windowWidth/8.5)
        imgui.create_context()

        ## Initialize GLFW
        if not glfw.init():
            raise Exception("GLFW initialization failed")

        ## Create a GLFW window
        self.window = glfw.create_window(self.windowWidth, self.windowHeigh, "OpenGL Window", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("GLFW window creation failed")

        ## Make the window's context current
        glfw.make_context_current(self.window)

        ## disable vsync
        glfw.swap_interval(0)

        self.impl = GlfwRenderer(self.window)

        ## Set the callback function for window resize
        glfw.set_framebuffer_size_callback(self.window, framebuffer_size_callback)
        
        ## Create and compile the vertex shader
        vertex_shader = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vertex_shader, self.VERTEX_SHADER)
        glCompileShader(vertex_shader)
        ## Create and compile the fragment shader
        fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(fragment_shader, self.FRAGMENT_SHADER)
        glCompileShader(fragment_shader)
        ## Create the shader program and link the shaders
        self.shader_program = glCreateProgram()
        glAttachShader(self.shader_program, vertex_shader)
        glAttachShader(self.shader_program, fragment_shader)
        glLinkProgram(self.shader_program)
        ## Delete the shaders (they are no longer needed)
        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)

        ## find uniform locations
        self.uniform_location_size = glGetUniformLocation(self.shader_program, "uSize")
        self.uniform_location_size_data = glGetUniformLocation(self.shader_program, "uSizeData")
        self.uniform_location_locx = glGetUniformLocation(self.shader_program, "uLocx")
        self.uniform_location_locy = glGetUniformLocation(self.shader_program, "uLocy")

        #!!! Genarate texture that taxture will have cuda tensor values
        self.vao = glGenVertexArrays(1)
        color = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, color)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        if self.elementSize == 2:
            glTexImage2D(GL_TEXTURE_2D,0, GL_R16F, self.tensorWidth, self.tensorHeight, 0, GL_RED, GL_FLOAT, None)
        elif self.elementSize == 4:
            glTexImage2D(GL_TEXTURE_2D,0, GL_R32F, self.tensorWidth, self.tensorHeight, 0, GL_RED, GL_FLOAT, None)
        glBindTexture(GL_TEXTURE_2D, 0)

        err, *_ = cu.cudaGLGetDevices(1, cu.cudaGLDeviceList.cudaGLDeviceListAll)
        # !!! Register
        err, self.cuda_image = cu.cudaGraphicsGLRegisterImage(color, GL_TEXTURE_2D, cu.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard,)
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)


        ## maximize at start
        # glfw.maximize_window(window)

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5*4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5*4, ctypes.c_void_p(0+3*4))
        glBindTexture(GL_TEXTURE_2D,color)
        # glBindVertexArray(0)
        
        ## points shape change from square to circle
        glEnable(GL_PROGRAM_POINT_SIZE)
        glEnable(GL_POINT_SMOOTH)
        self.set_enable_smooth = True

        glUseProgram(self.shader_program)
        if self.tensorHeight*self.tensorWidth<=100:
            glUniform1f(self.uniform_location_size, 50.0)
        elif self.tensorHeight*self.tensorWidth<=200:
            glUniform1f(self.uniform_location_size, 25.0)
        elif self.tensorHeight*self.tensorWidth<=10000:
            glUniform1f(self.uniform_location_size, 5.0)
        else:
            glUniform1f(self.uniform_location_size, 0.25)
            self.set_enable_smooth = False
        glUniform1f(self.uniform_location_size_data, self.windowWidth/8.5)

        ## Handle W A S D on keyboard for selection
        def keyaction(window, key, scancode, action, mods):
            if key == glfw.KEY_W and (action == glfw.PRESS or action == glfw.REPEAT): self.selectedY += 1
            if key == glfw.KEY_S and (action == glfw.PRESS or action == glfw.REPEAT): self.selectedY -= 1
            if key == glfw.KEY_D and (action == glfw.PRESS or action == glfw.REPEAT): self.selectedX += 1
            if key == glfw.KEY_A and (action == glfw.PRESS or action == glfw.REPEAT): self.selectedX -=1
        glfw.set_key_callback(self.window,keyaction)
        
        # glClearColor(0.05, 0.05, 0.1, 1.0)
        glClearColor(0.15, 0.16, 0.21, 1.0)
        ## Render loop
        lastTime = time.time()
        frameNumber = 0
        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            self.impl.process_inputs()
            imgui.new_frame()
            # imgui_render(self.tensor,self.tensorWidth,self.tensorHeight,self.uniform_location_locx,self.uniform_location_locy)
            self.imguiUI()
            glClear(GL_COLOR_BUFFER_BIT)
            currentTime = time.time()
            timeDiff = currentTime - lastTime
            frameNumber += 1
            # tensor += 0.0001
            if timeDiff >= 1.0 / 10.0:
                glfw.set_window_title(self.window, "FPS: "+str(int((1.0 / timeDiff) * frameNumber)))
                frameNumber = 0
                lastTime = currentTime
            (err,) = cu.cudaGraphicsMapResources(1, self.cuda_image, cu.cudaStreamLegacy)
            err, array = cu.cudaGraphicsSubResourceGetMappedArray(self.cuda_image, 0, 0)
            (err,) = cu.cudaMemcpy2DToArrayAsync(
                array,
                0,
                0,
                self.tensor.data_ptr(),
                self.elementSize*self.tensorWidth,
                self.elementSize*self.tensorWidth,
                self.tensorHeight,
                cu.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                cu.cudaStreamLegacy,
            )
            (err,) = cu.cudaGraphicsUnmapResources(1, self.cuda_image, cu.cudaStreamLegacy)

            glBindVertexArray(self.vao)
            if self.selectedX != -1 and self.selectedY != -1: 
                # glDrawArrays(GL_POINTS, 0, tensorHeight*tensorWidth+1)
                ## for big number like million if points will be like square specify more details 
                ## last vertex show value of seleceted index and show that in circle shape
                if self.set_enable_smooth:
                    glEnable(GL_POINT_SMOOTH)
                else:
                    glDisable(GL_POINT_SMOOTH)
                glDrawArrays(GL_POINTS, 0, self.tensorHeight*self.tensorWidth)
                glEnable(GL_POINT_SMOOTH)
                glDrawArrays(GL_POINTS, self.tensorHeight*self.tensorWidth,1)
            else: 
                glDrawArrays(GL_POINTS, 0, self.tensorHeight*self.tensorWidth)
            imgui.render()
            self.impl.render(imgui.get_draw_data())
            ## Swap buffers and poll events
            glfw.swap_buffers(self.window)
            ## Cleanup
        glDeleteProgram(self.shader_program)
        glDeleteBuffers(1, [self.vbo])
        self.impl.shutdown()
        glfw.terminate()

numpyArray = np.array([[0.1, 0.2, 0.3 ],
                       [0.4, 0.5, 0.6],
                       [0.7, 0.8, 0.9],
                       [1.0, 0.9, 0.8],])
tensor = torch.tensor(numpyArray,
                      dtype=torch.float16,
                      device=torch.device('cuda:0'))
GUI(tensor).renderOpenGL()


## example 2
# numpyArray=np.random.uniform(-0.5,1.5,(1000,1000))
# tensor = torch.tensor(numpyArray,
#                       dtype=torch.float32,
#                       device=torch.device('cuda:0'))
# GUI(tensor).renderOpenGL()