#include "./core/types.h"
#include "./core/maths.h"
#include "./core/platform.h"
#include "./core/mesh.h"
#include "../3rd/SDL2-2.0.4/include/SDL.h"
#include "../bridge.h"

#include <iostream>
#include <map>

#include "shaders.h"
#include "imgui.h"

#include "shadersDemoContext.h"
Mesh* g_toolMesh;
Mesh* g_originToolMesh;

float g_sphereRadius = 0.1;
int g_graphics = 0;	// 0=ogl, 1=DX11, 2=DX12
SDL_Window* g_window;			// window handle
unsigned int g_windowId;		// window id
Vec3 g_sceneLower = Vec3(-10);
Vec3 g_sceneUpper = Vec3(10);
int g_screenWidth = 1280;
int g_screenHeight = 720;
int g_msaaSamples = 8;
float g_fps;
float g_hapticFPS;
float qg_time_in_ms;
float tri_time_in_ms;

float g_planes[8][4];	
int g_numPlane = 1;

float g_cylinderLength = 30;

Vec3 g_camPos(6.0f, 8.0f, 18.0f);
Vec3 g_camAngle(0.0f, -DegToRad(20.0f), 0.0f);
Vec3 g_camVel(0.0f);
Vec3 g_camSmoothVel(0.0f);

Vec3 g_physicalToolPos(0.0f);
Vec3 g_physicalToolVel(0.0f);
float physicalToolSpeed = 0.1;
Vec3 g_virtualToolPos(0.0f);
Matrix44 g_hapticToolTrans;
Matrix44 g_virtualToolTrans;

float g_camSpeed;
float g_camNear;
float g_camFar;

Vec3 g_lightPos;
Vec3 g_lightDir;
Vec3 g_lightTarget;
std::vector<int> g_particleIdx;
FluidRenderer* g_fluidRenderer;
FluidRenderBuffers* g_fluidRenderBuffers;
DiffuseRenderBuffers* g_diffuseRenderBuffers;

ShadowMap* g_shadowMap;

bool g_fullscreen = false;
unsigned int g_frame = 0;
bool g_pause = false;
bool g_drawPoints = false;
bool g_drawSurfaceMesh = true;
bool g_drawMesh;
bool g_wireframe = false;
bool g_drawDensity = true;
bool g_showPanel= true;

using namespace std;

enum DISPLAYMODEVALUE {VOLUMN, COLLISION};
char* displayModeName[2] = {"VOLUMN", "COLLISION"};
int g_forceDisplayModeNum = 2;
int g_displayMode = 0;

int g_numSubsteps;

char g_deviceName[256];
bool g_vsync = true;

bool g_benchmark = false;
bool g_extensions = true;
bool g_teamCity = false;
bool g_interop = true;
bool g_d3d12 = false;
bool g_useAsyncCompute = true;		
bool g_increaseGfxLoadForAsyncComputeTesting = false;

Vec4 g_fluidColor;
Vec4 g_diffuseColor;
Vec3 g_meshColor;
Vec3  g_clearColor;
float g_lightDistance;
float g_fogDistance;

int g_lastx;
int g_lasty;
int g_lastb = -1;
int g_drawSprings;


float g_waitTime;		// the CPU time spent waiting for the GPU
float g_updateTime;     // the CPU time spent on Flex
float g_renderTime;		// the CPU time spent calling OpenGL to render the scene
						// the above times don't include waiting for vsync
float g_simLatency;     // the time the GPU spent between the first and last NvFlexUpdateSolver() operation. Because some GPUs context switch, this can include graphics time.

int g_levelScroll;			// offset for level selection scroll area

int g_mouseParticle = -1;
float g_mouseT = 0.0f;
Vec3 g_mousePos;
float g_mouseMass;
bool g_mousePicked = false;

inline float sqr(float x) { return x * x; }

/* Note that this array of colors is altered by demo code, and is also read from global by graphics API impls */
Colour g_colors[] =
{
	Colour(0.0f, 0.5f, 1.0f),
	Colour(0.797f, 0.354f, 0.000f),
	Colour(0.092f, 0.465f, 0.820f),
	Colour(0.000f, 0.349f, 0.173f),
	Colour(0.875f, 0.782f, 0.051f),
	Colour(0.000f, 0.170f, 0.453f),
	Colour(0.673f, 0.111f, 0.000f),
	Colour(0.612f, 0.194f, 0.394f)
};


Matrix44 g_camBase;
Matrix44 g_camTrans;
Matrix44 g_camTipTrans = Matrix44::kIdentity;
Point3 g_camBiDir;
Vec3 g_toolCamAngle;
int g_stepNum;

void Draw3DMesh(float* vert, float* norm, float* color, unsigned int* tri,   int triNum) {
	DrawMesh(vert, norm, color, tri, triNum);
}

void AddFrameCount() {
	g_frame++;
}

void SetEndoscopePos(float* trans) {
	memcpy(trans, &g_camTrans[0], 16 * sizeof(float));
}


void DrawShapes() {};

void DrawImguiString(int x, int y, Vec3 color, int align, const char* s, ...)
{
	char buf[2048];

	va_list args;

	va_start(args, s);
	vsnprintf(buf, 2048, s, args);
	va_end(args);

	imguiDrawText(x, y, align, buf, imguiRGBA((unsigned char)(color.x * 255), (unsigned char)(color.y * 255), (unsigned char)(color.z * 255)));
}





void Init(bool centerCamera = true)
{
	RandInit();
	g_fluidColor = Vec4(0.1f, 0.4f, 0.8f, 1.0f);
	g_meshColor = Vec3(0.9f, 0.9f, 0.9f);
	g_drawPoints = false;
	g_drawSprings = false;
	g_drawMesh = false;
	g_lightDistance = 2.0f;
	g_fogDistance = 0.005f;

	g_camSpeed = 0.075f;
	g_camNear = 0.01f;
	g_camFar = 1000.0f;

	// center camera on particles
	if (centerCamera)
	{
		g_camPos = Vec3(
			(g_sceneLower.x + g_sceneUpper.x)*0.5f, 
			(g_sceneLower.y + g_sceneUpper.y) * 0.5f,
			g_sceneUpper.z + min(g_sceneUpper.y, 6.0f)*2.0f);
		g_camAngle = Vec3(0.0f, -DegToRad(15.0f), 0.0f);
	}
	int tetVertNum = GetPointsNum();
	cout << "tetVertNum: " << tetVertNum << endl;
	g_particleIdx.resize(tetVertNum);
	for (int i = 0; i < tetVertNum; i++)
		g_particleIdx[i] = i;
	g_fluidRenderBuffers = CreateFluidRenderBuffers(tetVertNum, g_interop);
	g_diffuseRenderBuffers = CreateDiffuseRenderBuffers(tetVertNum, g_interop);

}

void UpdateCamera()
{
	Vec3 forward(-sinf(g_camAngle.x)*cosf(g_camAngle.y), sinf(g_camAngle.y), -cosf(g_camAngle.x)*cosf(g_camAngle.y));
	Vec3 right(Normalize(Cross(forward, Vec3(0.0f, 1.0f, 0.0f))));

	g_camSmoothVel = Lerp(g_camSmoothVel, g_camVel, 0.1f);
	g_camPos += (forward*g_camSmoothVel.z + right*g_camSmoothVel.x + Cross(right, forward)*g_camSmoothVel.y);
}

void UpdatePhysicalTool()
{
	if (!g_useHapticDevice)
	{
		g_physicalToolPos += g_physicalToolVel;
		bridge_qh[0] = g_physicalToolPos.x;
		bridge_qh[1] = g_physicalToolPos.y;
		bridge_qh[2] = g_physicalToolPos.z;
	}
	else
	{
		g_physicalToolPos.x = bridge_qh[0];
		g_physicalToolPos.y = bridge_qh[1];
		g_physicalToolPos.z = bridge_qh[2];
		g_hapticToolTrans = Matrix44(hapticToolTrans);
		g_virtualToolTrans = Matrix44(virtualToolTrans);
		//cout << "col 0:" << hapticToolTrans[0] << " " << hapticToolTrans[1] << " " << hapticToolTrans[2] << " " << hapticToolTrans[3]<<endl
		//	<< " " << hapticToolTrans[4] << " " << hapticToolTrans[5] << " " << hapticToolTrans[6] << " " << hapticToolTrans[7] << endl
		//	<< " " << hapticToolTrans[8] << " " << hapticToolTrans[9] << " " << hapticToolTrans[10] << " " << hapticToolTrans[11] << endl
		//	<< " " << hapticToolTrans[12] << " " << hapticToolTrans[13] << " " << hapticToolTrans[14] << " " << hapticToolTrans[15] << endl;
		//auto pos = g_hapticToolTrans.GetCol(3);
		//auto vec_0 = g_hapticToolTrans.GetCol(0);
		//auto vec_1 = g_hapticToolTrans.GetCol(1);
		//auto vec_2 = g_hapticToolTrans.GetCol(2);
		//cout << "hapticToolTrans col4:\n" << pos.x<<" "<<pos.y<<" "<<pos.z<<" "<<pos.w << endl;
		//cout << vec_0.x << " " << vec_0.y << " " << vec_0.z << " " << vec_0.w << endl;
		//cout << vec_1.x << " " << vec_1.y << " " << vec_1.z << " " << vec_1.w << endl;
		//cout << vec_2.x << " " << vec_2.y << " " << vec_2.z << " " << vec_2.w << endl;
	}
}

void UpdateVirtualTool()
{
	g_virtualToolPos.x = bridge_qg[0];
	g_virtualToolPos.y = bridge_qg[1];
	g_virtualToolPos.z = bridge_qg[2];
}

void RenderScene()
{
	const int numParticles = GetPointsNum();
#pragma region 更新粒子位置
	if (g_drawPoints)
	{
		//UpdateFluidRenderBuffers(g_fluidRenderBuffers,GetPointsPtr(), numParticles, g_particleIdx.data(), g_particleIdx.size());
		UpdateFluidRenderBuffers(g_fluidRenderBuffers, GetPointsPtr(), numParticles, GetForceIntensityPtr(g_displayMode), g_particleIdx.data(), g_particleIdx.size());
	}
#pragma endregion

#pragma region 计算相机灯光位置

	float fov = kPi / 4.0f;
	float aspect = float(g_screenWidth) / g_screenHeight;

	Matrix44 proj = ProjectionMatrix(RadToDeg(fov), aspect, g_camNear, g_camFar);
	Matrix44 view;
		view = RotationMatrix(-g_camAngle.x,
		Vec3(0.0f, 1.0f, 0.0f))*RotationMatrix(-g_camAngle.y, Vec3(cosf(-g_camAngle.x), 0.0f, sinf(-g_camAngle.x)))*TranslationMatrix(-Point3(g_camPos));

	// expand scene bounds to fit most scenes
	g_sceneLower = Min(g_sceneLower, Vec3(-2.0f, 0.0f, -2.0f));
	g_sceneUpper = Max(g_sceneUpper, Vec3(2.0f, 2.0f, 2.0f));

	Vec3 sceneExtents = g_sceneUpper - g_sceneLower;
	Vec3 sceneCenter = 0.5f*(g_sceneUpper + g_sceneLower);

	g_lightDir = Normalize(Vec3(5.0f, 15.0f, 7.5f));
	g_lightPos = sceneCenter + g_lightDir*Length(sceneExtents)*g_lightDistance;
	g_lightTarget = sceneCenter;

	// calculate tight bounds for shadow frustum
	float lightFov = 2.0f*atanf(Length(g_sceneUpper - sceneCenter) / Length(g_lightPos - sceneCenter));

	// scale and clamp fov for aesthetics
	lightFov = Clamp(lightFov, DegToRad(25.0f), DegToRad(65.0f));

	Matrix44 lightPerspective = ProjectionMatrix(RadToDeg(lightFov), 1.0f, 1.0f, 1000.0f);
	Matrix44 lightView = LookAtMatrix(Point3(g_lightPos), Point3(g_lightTarget));
	Matrix44 lightTransform = lightPerspective*lightView;
#pragma endregion

	Matrix44 toolTrans = g_hapticToolTrans;
	Mesh* hapticToolMesh = new Mesh(*g_toolMesh);
	Mesh* virtualToolMesh = new Mesh(*g_toolMesh);
	hapticToolMesh->Transform(toolTrans);

	Matrix44 VTTrans = g_virtualToolTrans;
	VTTrans.SetCol(3, Vector4(g_virtualToolPos.x, g_virtualToolPos.y, g_virtualToolPos.z, 1.0f));
	virtualToolMesh->Transform(VTTrans);

#pragma region 计算阴影

	ShadowBegin(g_shadowMap);

	SetView(lightView, lightPerspective);
	SetCullMode(false);
	//Draw3DScene();
	ShadowEnd();
#pragma endregion 

#pragma region 计算光照

	BindSolidShader(g_lightPos, g_lightTarget, lightTransform, g_shadowMap, 0.0f, Vec4(g_clearColor, g_fogDistance));

	SetView(view, proj);
	SetCullMode(true);

	if(g_drawSurfaceMesh)
		Draw3DScene();

	DrawMesh(hapticToolMesh, Vec3(0.1, 0.8, 0.8));
	DrawMesh(virtualToolMesh, Vec3(0.5));

	DrawPlanes((Vec4*)g_planes, g_numPlane, 0);

	UnbindSolidShader();
	delete hapticToolMesh;
	delete virtualToolMesh;

	if (g_drawPoints)
		DrawPoints(g_fluidRenderBuffers, numParticles, 0, g_sphereRadius, float(g_screenWidth), aspect, fov, 
			g_lightPos, g_lightTarget, lightTransform, g_shadowMap, g_drawDensity);
#pragma endregion
}

void RenderDebug()
{
	
}


// returns the new scene if one is selected
void DoUI()
{
	if (!g_showPanel)
		return;
	int x = g_screenWidth - 200;
	int y = g_screenHeight - 23;
	int uiOffset = 250;
	int uiBorder = 20;
	int uiWidth = 200;
	int uiHeight = g_screenHeight - uiOffset - uiBorder * 3;
	int uiLeft = uiBorder;

	unsigned char button = 0;
	if (g_lastb == SDL_BUTTON_LEFT)
		button = IMGUI_MBUT_LEFT;
	else if (g_lastb == SDL_BUTTON_RIGHT)
		button = IMGUI_MBUT_RIGHT;
	imguiBeginFrame(g_lastx, g_screenHeight - g_lasty, button, 0);
	x += 180;


	DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Frame: %d", g_frame);
	y -= 20;
	DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Fps: %.2f", g_fps);
	y -= 20;
	DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "haptic Fps: %.2f", g_hapticFPS);
	y -= 20;
	DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "qg time: %.3f ms", qg_time_in_ms);
	y -= 20;
	DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "tri time: %.3f ms", tri_time_in_ms);

	static int scroll = 0;
	imguiBeginScrollArea("Options", uiLeft, g_screenHeight - uiBorder - uiHeight - uiOffset - uiBorder, uiWidth, uiHeight, &scroll);
	if (imguiCheck("Pause", g_pause))
		g_pause = !g_pause;

	if (imguiCheck("DrawSurfaceMesh", g_drawSurfaceMesh))
		g_drawSurfaceMesh = !g_drawSurfaceMesh;
	if (imguiCheck("DrawPoints", g_drawPoints))
		g_drawPoints = !g_drawPoints;
	if (imguiButton(displayModeName[g_displayMode]))
	{
		g_displayMode = (g_displayMode + 1) % g_forceDisplayModeNum;
	}


	if(g_drawPoints)
		imguiSlider("Solid Radius", &g_sphereRadius, 0.0f, 0.5f, 0.001f);

	float gx = GetGravityX();
	if (imguiSlider("GravityX", &gx, -9.8, 9.8, 0.5))
		SetGravityX(gx);
	float gy = GetGravityY();
	if (imguiSlider("GravityY", &gy, -9.8, 9.8, 0.5))
		SetGravityY(gy);
	float gz = GetGravityZ();
	if (imguiSlider("GravityZ", &gz, -9.8, 9.8, 0.5))
		SetGravityZ(gz);


	imguiEndScrollArea();
	imguiEndFrame();
	DrawImguiGraph();


}

void UpdateFrame()
{
	g_fps = 1.0f/GetSimTime();
	g_hapticFPS = 1.0f / GetHapticTime();
	qg_time_in_ms = GetQGTime() * 1000;
	tri_time_in_ms = GetTriTime() * 1000;
	// Getting timers causes CPU/GPU sync, so we do it after a map

	UpdateCamera();
	UpdatePhysicalTool();
	UpdateVirtualTool();

	double renderBeginTime = GetSeconds();

	StartFrame(Vec4(g_clearColor, 1.0f));

	// main scene render
	RenderScene();
	RenderDebug();

	DoUI();

	EndFrame();

	// move mouse particle (must be done here as GetViewRay() uses the GL projection state)
	if (g_mouseParticle != -1)
	{
		Vec3 origin, dir;
		GetViewRay(g_lastx, g_screenHeight - g_lasty, origin, dir);

		g_mousePos = origin + dir*g_mouseT;
	}

		
	double renderEndTime = GetSeconds();

	// Exponential filter to make the display easier to read
	//const float timerSmoothing = 0.05f;

	//g_updateTime = (g_updateTime == 0.0f) ? newUpdateTime : Lerp(g_updateTime, newUpdateTime, timerSmoothing);
	//g_renderTime = (g_renderTime == 0.0f) ? newRenderTime : Lerp(g_renderTime, newRenderTime, timerSmoothing);
	//g_waitTime = (g_waitTime == 0.0f) ? newWaitTime : Lerp(g_waitTime, newWaitTime, timerSmoothing);

	PresentFrame(g_vsync);

}


void ReshapeWindow(int width, int height)
{
	if (!g_benchmark)
		printf("Reshaping\n");

	ReshapeRender(g_window);

	if (!g_fluidRenderer || (width != g_screenWidth || height != g_screenHeight))
	{
		if (g_fluidRenderer)
			DestroyFluidRenderer(g_fluidRenderer);
		g_fluidRenderer = CreateFluidRenderer(width, height);
	}

	g_screenWidth = width;
	g_screenHeight = height;
}

void InputArrowKeysDown(int key, int x, int y)
{
	switch (key)
	{
	case SDLK_DOWN:
	{
		// update scroll UI to center on selected scene
		break;
	}
	case SDLK_UP:
	{

		// update scroll UI to center on selected scene
		break;
	}
	case SDLK_LEFT:
	{

		// update scroll UI to center on selected scene
		break;
	}
	case SDLK_RIGHT:
	{
		// update scroll UI to center on selected scene

		break;
	}
	}
}

void InputArrowKeysUp(int key, int x, int y)
{
}

bool InputKeyboardDown(unsigned char key, int x, int y)
{
	printf("keydown\n");
	if (key > '0' && key <= '9')
	{

		return false;
	}

	float kSpeed = g_camSpeed;

	switch (key)
	{
	case 'w':
	{
		g_camVel.z = kSpeed;
		break;
	}
	case 's':
	{
		g_camVel.z = -kSpeed;
		break;
	}
	case 'a':
	{
		g_camVel.x = -kSpeed;
		break;
	}
	case 'd':
	{
		g_camVel.x = kSpeed;
		break;
	}
	case 'q':
	{
		g_camVel.y = kSpeed;
		break;
	}
	case 'z':
	{
		//g_drawCloth = !g_drawCloth;
		g_camVel.y = -kSpeed;
		break;
	}

	case 'u':
	{
		if (g_fullscreen)
		{
			SDL_SetWindowFullscreen(g_window, 0);
			ReshapeWindow(1280, 720);
			g_fullscreen = false;
		}
		else
		{
			SDL_SetWindowFullscreen(g_window, SDL_WINDOW_FULLSCREEN_DESKTOP);
			g_fullscreen = true;
		}
		break;
	}
	case 'r':
	{
		ResetXg(g_physicalToolPos.x, g_physicalToolPos.y, g_physicalToolPos.z);
		break;
	}
	case 'y':
	{
		break;
	}
	case 'p':
	{
		g_pause = !g_pause;
		break;
	}
	case 'h':
	{
		g_showPanel = !g_showPanel;
		break;
	}
	case 'e':
	{

		break;
	}
	case 't':
	{

		break;
	}
	case 'v':
	{
		g_drawPoints = !g_drawPoints;
		break;
	}
	case 'f':
	{
		g_drawSprings = (g_drawSprings + 1) % 3;
		break;
	}

	case 'm':
	{
		g_drawMesh = !g_drawMesh;
		break;
	}

	case 'j':
	{
		g_physicalToolVel.x = -physicalToolSpeed;
		break;
	}
	case 'l':
	{
		g_physicalToolVel.x = physicalToolSpeed;
		break;
	}
	case 'i':
	{
		g_physicalToolVel.y = physicalToolSpeed;
		break;
	}
	case 'k':
	{
		g_physicalToolVel.y = -physicalToolSpeed;
		break;
	}
	case 'o':
	{
		g_physicalToolVel.z = -physicalToolSpeed;
		break;
	}
	case ',':
	{
		g_physicalToolVel.z = physicalToolSpeed;
		break;
	}
	case ' ':
	{
		break;
	}
	case ';':
	{
		break;
	}
	case 13:
	{
		break;
	}
	case 27:
	{
		// return quit = true
		return true;
	}
	};


	return false;
}

void InputKeyboardUp(unsigned char key, int x, int y)
{
	switch (key)
	{
	case 'w':
	case 's':
	{
		g_camVel.z = 0.0f;
		break;
	}
	case 'a':
	case 'd':
	{
		g_camVel.x = 0.0f;
		break;
	}
	case 'q':
	case 'z':
	{
		g_camVel.y = 0.0f;
		break;
	}
	case 'j':
	case 'l':
	{
		g_physicalToolVel.x = 0.0f;
		break;
	}
	case 'k':
	case 'i':
	{
		g_physicalToolVel.y = 0.0f;
		break;
	}
	case 'o':
	case ',':
	{
		g_physicalToolVel.z = 0.0f;
		break;
	}
	};
}

void MouseFunc(int b, int state, int x, int y)
{
	switch (state)
	{
	case SDL_RELEASED:
	{
		g_lastx = x;
		g_lasty = y;
		g_lastb = -1;

		break;
	}
	case SDL_PRESSED:
	{
		g_lastx = x;
		g_lasty = y;
		g_lastb = b;
#ifdef ANDROID
		extern void setStateLeft(bool bLeftDown);
		setStateLeft(false);
#else
		if ((SDL_GetModState() & KMOD_LSHIFT) && g_lastb == SDL_BUTTON_LEFT)
		{
			// record that we need to update the picked particle
			g_mousePicked = true;
		}
#endif
		break;
	}
	};
}

void MousePassiveMotionFunc(int x, int y)
{
	g_lastx = x;
	g_lasty = y;
}

void MouseMotionFunc(unsigned state, int x, int y)
{
	float dx = float(x - g_lastx);
	float dy = float(y - g_lasty);

	g_lastx = x;
	g_lasty = y;

	if (state & SDL_BUTTON_RMASK)
	{
		const float kSensitivity = DegToRad(0.1f);
		const float kMaxDelta = FLT_MAX;

		g_camAngle.x -= Clamp(dx*kSensitivity, -kMaxDelta, kMaxDelta);
		g_camAngle.y -= Clamp(dy*kSensitivity, -kMaxDelta, kMaxDelta);
	}
}


void SDLInit(const char* title)
{
	if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMECONTROLLER) < 0)	// Initialize SDL's Video subsystem and game controllers
		printf("Unable to initialize SDL");

	unsigned int flags = SDL_WINDOW_RESIZABLE;
	if (g_graphics == 0)
	{
		SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
		flags = SDL_WINDOW_RESIZABLE | SDL_WINDOW_OPENGL;
	}


	g_window = SDL_CreateWindow(title, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
		g_screenWidth, g_screenHeight, flags);

	g_windowId = SDL_GetWindowID(g_window);
}

bool GLUpdate()
{
#if ENABLE_AFTERMATH_SUPPORT
	__try
#endif
	{
		bool quit = true;
		SDL_Event e;
		UpdateFrame();

		while (SDL_PollEvent(&e))
		{
			switch (e.type)
			{
			case SDL_QUIT:
				quit = false;
				break;

			case SDL_KEYDOWN:
				if (e.key.keysym.sym < 256 && (e.key.keysym.mod == KMOD_NONE || (e.key.keysym.mod & KMOD_NUM)))
					InputKeyboardDown(e.key.keysym.sym, 0, 0);
				InputArrowKeysDown(e.key.keysym.sym, 0, 0);
				break;

			case SDL_KEYUP:
				if (e.key.keysym.sym < 256 && (e.key.keysym.mod == 0 || (e.key.keysym.mod & KMOD_NUM)))
					InputKeyboardUp(e.key.keysym.sym, 0, 0);
				InputArrowKeysUp(e.key.keysym.sym, 0, 0);
				break;

			case SDL_MOUSEMOTION:
				if (e.motion.state)
					MouseMotionFunc(e.motion.state, e.motion.x, e.motion.y);
				else
					MousePassiveMotionFunc(e.motion.x, e.motion.y);
				break;

			case SDL_MOUSEBUTTONDOWN:
			case SDL_MOUSEBUTTONUP:
				MouseFunc(e.button.button, e.button.state, e.motion.x, e.motion.y);
				break;

			case SDL_WINDOWEVENT:
				if (e.window.windowID == g_windowId)
				{
					if (e.window.event == SDL_WINDOWEVENT_SIZE_CHANGED)
						ReshapeWindow(e.window.data1, e.window.data2);
				}
				break;

			case SDL_WINDOWEVENT_LEAVE:
				g_camVel = Vec3(0.0f, 0.0f, 0.0f);
				break;
			}
		}
		return quit;
	}

}

void GLInit()
{
	g_toolMesh = ImportMesh(".\\data\\CoarseToothedGripper_Pivot.obj");

	float region[16];

	GetRegion(region);
	g_sceneLower.x = region[0];
	g_sceneLower.y = region[1];
	g_sceneLower.z = region[2];
	g_sceneUpper.x = region[3];
	g_sceneUpper.y = region[4];
	g_sceneUpper.z = region[5];

	(Vec4&)g_planes[0] = Vec4(0.0f, 1.0f, 0.0f, -g_sceneLower.y);
	RenderInitOptions options;
	CreateDemoContext(g_graphics);

	SDLInit("Render Test");

	options.window = g_window;
	options.numMsaaSamples = g_msaaSamples;
	options.asyncComputeBenchmark = false;
	options.defaultFontHeight = -1;
	options.fullscreen = g_fullscreen;

	InitRender(options);

	if (g_fullscreen)
		SDL_SetWindowFullscreen(g_window, SDL_WINDOW_FULLSCREEN_DESKTOP);

	ReshapeWindow(g_screenWidth, g_screenHeight);
	// create shadow maps
	g_shadowMap = ShadowCreate();
	Init();
}

void GLDestroy() {
	if (g_fluidRenderer)
		DestroyFluidRenderer(g_fluidRenderer);

	if(g_fluidRenderBuffers)
		DestroyFluidRenderBuffers(g_fluidRenderBuffers);

	if(g_diffuseRenderBuffers)
		DestroyDiffuseRenderBuffers(g_diffuseRenderBuffers);

	if(g_shadowMap)
		ShadowDestroy(g_shadowMap);

	DestroyRender();

	if(g_window)
		SDL_DestroyWindow(g_window);
	SDL_Quit();

}
