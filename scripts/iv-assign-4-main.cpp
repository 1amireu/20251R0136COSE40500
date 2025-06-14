/*
 * Skeleton code for COSE436 Fall 2024
 *
 * Won-Ki Jeong, wkjeong@korea.ac.kr
 *
 */

#include <stdio.h>
#include <GL/glew.h>
#include <GL/glut.h>

#include <iostream>
#include <assert.h>
#include "textfile.h"
#include "tfeditor.h"
#include "Angel.h"
/*  
 #define FILE_NAME "../data/CThead_512_512_452.raw"
 #define W 512
 #define H 512
 #define D 452
*/

 #define FILE_NAME "../data/tooth_100_90_160.raw"
 #define W 100
 #define H 90
 #define D 160
 
 /*
 #define FILE_NAME "../data/Bucky_32_32_32.raw"
 #define W 32
 #define H 32
 #define D 32
 */
 /*
 #define FILE_NAME "../data/bonsai_256_256_256.raw"
 #define W 256
 #define H 256
 #define D 256
 */
/*
#define FILE_NAME "../data/lung_256_256_128.raw"
#define W 256
#define H 256
#define D 128
 */
mat4 rotate(float angle, const vec3& axis);

// Glut windows
int volumeRenderingWindow;
int transferFunctionWindow;

// Shader programs
GLuint p;

// Texture object
GLuint objectTex;

GLuint transferFunctionTexture;

float stepSize = 0.01;
float isoValue = 0.5;
float alphaPower = 1.0;

//
// Loading volume file, create 3D texture and its histogram
//
void load3Dfile(char* filename, int w, int h, int d) {

	// loading volume data
	FILE* f = fopen(filename, "rb");
	unsigned char* data = new unsigned char[w * h * d];
	fread(data, 1, w * h * d, f);
	fclose(f);

	// generate 3D texture
	glGenTextures(1, &objectTex);
	glBindTexture(GL_TEXTURE_3D, objectTex);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

	glTexImage3D(GL_TEXTURE_3D, 0, GL_RED, w, h, d, 0, GL_RED, GL_UNSIGNED_BYTE, data);

	// create histogram
	for (int i = 0; i < 256; i++) {
		histogram[i] = 0;
	}
	for (int i = 0; i < w * h * d; i++) {
		histogram[data[i]]++;
	}
	for (int i = 0; i < 256; i++) {
		histogram[i] /= w * h * d;
	}

	delete[]data;
}


void changeSize(int w, int h) {

	// Prevent a divide by zero, when window is too short
	// (you cant make a window of zero width).
	if (h == 0) h = 1;
	float ratio = 1.0f * (float)w / (float)h;

	// Set the viewport to be the entire window
	glViewport(0, 0, w, h);
}

float tx = 0.0, ty = 0.0, tz = 0.0;

void keyboard(unsigned char key, int x, int y)
{
	bool centChanged = false;
	if (key == '1') glUniform1i(glGetUniformLocation(p, "renderingMode"), 1); // MIP
	if (key == '2') glUniform1i(glGetUniformLocation(p, "renderingMode"), 2); // Isosurface
	if (key == '3') glUniform1i(glGetUniformLocation(p, "renderingMode"), 3); // Alpha compositing
	if (key == '-') {
		alphaPower -= 0.01;
		glUniform1f(glGetUniformLocation(p, "alphaPower"), alphaPower);
	}
	if (key == '+') {
		alphaPower += 0.01;
		glUniform1f(glGetUniformLocation(p, "alphaPower"), alphaPower);
	}
	glutPostRedisplay();
}

void specialKeys(int key, int x, int y) {
	if (key == GLUT_KEY_UP) {
		isoValue = isoValue + 0.001;
		glUniform1f(glGetUniformLocation(p, "isoValue"), isoValue);
		std::cout << "up" << std::endl;
	}
	if (key == GLUT_KEY_DOWN) {
		isoValue = isoValue - 0.001;
		if (isoValue <= 0.001) isoValue = 0.001;
		glUniform1f(glGetUniformLocation(p, "isoValue"), isoValue);
		std::cout << "down" << std::endl;
	}
	if (key == GLUT_KEY_RIGHT) {
		stepSize = stepSize + 0.001;
		glUniform1f(glGetUniformLocation(p, "stepSize"), stepSize);
		std::cout << "right" << std::endl;
	}
	if (key == GLUT_KEY_LEFT) {
		stepSize = stepSize - 0.001;
		if (stepSize <= 0.001) stepSize = 0.001;
		glUniform1f(glGetUniformLocation(p, "stepSize"), stepSize);
		std::cout << "left" << std::endl;
	}
	glutPostRedisplay();
}

// Virtual Trackball
GLfloat lastPos[3] = { 0.0f, 0.0f, 1.0f };
bool leftMouse = false, rightMouse = false;
vec3 rotationAxis = vec3(0.0f);
GLfloat zoomFactor = 1.0f, dragFactor = 3.0f;

// Look-At matrix
vec4 eye = { 0.0f, 0.0f, 1.0f, 1.0f };
vec4 at = { 0.0f, 0.0f, 0.0f, 1.0f };
vec4 up = { 1.0f, 1.0f, 1.0f, 0.0f };
mat4 viewMatrix = LookAt(eye, at, up);
vec4 cameraPosition = eye;
vec4 upDir = up;

float sliceDepth = 0.0f;  // Current slice depth (range 0 to 1)
float sliceSpeed = 0.01f;  // Speed at which slices change

void trackball_ptov(int x, int y, int width, int height, float v[3])
{
	GLfloat d, a;

	/* project x,y onto a hemi-sphere centered within width,height */
	v[0] = (2.0F * x - width) / width;
	v[1] = (height - 2.0F * y) / height;
	d = (GLfloat)sqrt(v[0] * v[0] + v[1] * v[1]);
	v[2] = (GLfloat)cos((M_PI / 2.0F) * ((d < 1.0F) ? d : 1.0F));
	a = 1.0F / (GLfloat)sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
	v[0] *= a;
	v[1] *= a;
}

// Mouse click handler to start/stop dragging
void mouseClick(int button, int state, int x, int y) {
	GLuint winWidth = glutGet(GLUT_WINDOW_WIDTH);
	GLuint winHeight = glutGet(GLUT_WINDOW_HEIGHT);

	if (button == GLUT_LEFT_BUTTON) {
		if (state == GLUT_DOWN) {
			leftMouse = true;  // Start rotating
			trackball_ptov(x, y, winWidth, winHeight, lastPos);
		}
		else if (state == GLUT_UP) {
			leftMouse = false;  // Stop rotating
		}
	}
	else if (button == GLUT_RIGHT_BUTTON) {
		if (state == GLUT_DOWN) {
			rightMouse = true;  // Start zooming
			lastPos[0] = x;
			lastPos[1] = y;
		}
		else if (state == GLUT_UP) {
			rightMouse = false;  // Stop zooming
		}
	}
}

mat4 computeTrackballRotation(float angle, vec3 axis, vec3 centerPosition) {
	// Translate object to the origin
	mat4 translateToOrigin = Translate(-centerPosition.x, -centerPosition.y, -centerPosition.z);

	// Rotate around the axis
	mat4 rotation = rotate(-angle, axis);

	// Translate object back to its original position
	mat4 translateBack = Translate(centerPosition.x, centerPosition.y, centerPosition.z);

	// Combine transformations
	return translateBack * rotation * translateToOrigin;
}

// Mouse motion handler
void mouseMove(int x, int y) {
	GLfloat curPos[3], dx, dy, dz;
	GLuint winWidth = glutGet(GLUT_WINDOW_WIDTH);
	GLuint winHeight = glutGet(GLUT_WINDOW_HEIGHT);

	if (leftMouse) {
		/* Handle rotation as before */
		trackball_ptov(x, y, winWidth, winHeight, curPos);
		dx = curPos[0] - lastPos[0];
		dy = curPos[1] - lastPos[1];
		dz = curPos[2] - lastPos[2];

		if (dx || dy || dz) {
			GLfloat angle = 90.0 * sqrt(dx * dx + dy * dy + dz * dz);
			rotationAxis[0] = lastPos[1] * curPos[2] - lastPos[2] * curPos[1];
			rotationAxis[1] = lastPos[2] * curPos[0] - lastPos[0] * curPos[2];
			rotationAxis[2] = lastPos[0] * curPos[1] - lastPos[1] * curPos[0];

			lastPos[0] = curPos[0];
			lastPos[1] = curPos[1];
			lastPos[2] = curPos[2];

			mat4 inverseView = transpose(viewMatrix);
			vec4 transformedAxis = inverseView * vec4(rotationAxis, 0.0f);
			rotationAxis = normalize(vec3(transformedAxis.x, transformedAxis.y, transformedAxis.z));

			// Calculate rotation matrix
			mat4 rotationMatrix = computeTrackballRotation(angle * M_PI /180, rotationAxis, vec3(0.5f, 0.5f, 0.5f));

			// Apply rotation to camera position and up direction
			cameraPosition = rotationMatrix * cameraPosition;
			upDir = rotationMatrix * upDir;

			// Update view matrix
			viewMatrix = LookAt(cameraPosition, vec3(0.5f, 0.5f, 0.5f), upDir);
		}
		glutPostRedisplay();
	}

	if (rightMouse) {
		float dy = y - lastPos[1];

		zoomFactor += dy * 0.01f;
		if (zoomFactor < 0) zoomFactor = 0.0f;

		lastPos[0] = x;
		lastPos[1] = y;

		glutPostRedisplay();
	}
}

void mouseWheel(int button, int dir, int x, int y) {
	GLfloat sliceFactor = dir * sliceSpeed;

	// Ensure slice depth stays within valid bounds
	if (sliceFactor < 0) {
		sliceDepth = std::max(0.0f, sliceDepth + sliceFactor);
	}
	else {
		sliceDepth = std::min(1.0f, sliceDepth + sliceFactor);
	}
	// Update the slice depth in the shader
	glUniform1f(glGetUniformLocation(p, "sliceDepth"), sliceDepth);

	// Request to redraw the scene
	glutPostRedisplay();
}

void renderScene(void)
{
	glClearColor(0, 0, 0, 0);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram(p);

	vec3 eyePosition = { cameraPosition.x, cameraPosition.y, cameraPosition.z };
	vec3 update = { upDir.x, upDir.y, upDir.z };

	glUniform3fv(glGetUniformLocation(p, "eyePosition"), 1, eyePosition);
	glUniform3fv(glGetUniformLocation(p, "up"), 1, update);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_1D, transferFunctionTexture);
	glUniform1i(glGetUniformLocation(p, "transferFunction"), 1);

	glUniform1f(glGetUniformLocation(p, "zoomFactor"), zoomFactor);

	// Draw a simple object (cube in this case)
	glBegin(GL_QUADS);

	// Define vertices of a cube
	glColor3f(1.0f, 0.0f, 0.0f);
	glVertex3f(-1.0f, -1.0f, -1.0f);  // Vertex 0
	glVertex3f(1.0f, -1.0f, -1.0f);  // Vertex 1
	glVertex3f(1.0f, 1.0f, -1.0f);  // Vertex 2
	glVertex3f(-1.0f, 1.0f, -1.0f);  // Vertex 3
	glEnd();

	// Debug transfer function values
	std::cout << "R,G,B,A: " << transferFunction[0] << "," << transferFunction[1] << "," << transferFunction[2] << "," << transferFunction[3] << std::endl;

	glutSwapBuffers();
}


void idle()
{
	if (transferFunctionChanged) {
		glutSetWindow(volumeRenderingWindow);
		glBindTexture(GL_TEXTURE_1D, transferFunctionTexture);
		glTexSubImage1D(GL_TEXTURE_1D, 0, 0, 256, GL_RGBA, GL_FLOAT, transferFunction);
		transferFunctionChanged = false;
		glutPostRedisplay();
	}
}


void init()
{
	load3Dfile(FILE_NAME, W, H, D);
	glUseProgram(p);

	// Set the texture unit
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_3D, objectTex);
	glUniform1i(glGetUniformLocation(p, "tex"), 0);

	// Set volume bounds
	glUniform3fv(glGetUniformLocation(p, "objectMin"), 1, vec3(0.0f));
	glUniform3fv(glGetUniformLocation(p, "objectMax"), 1, vec3(1.0f));

	glUniform1i(glGetUniformLocation(p, "renderingMode"), 1);

	glUniform1f(glGetUniformLocation(p, "stepSize"), stepSize);
	glUniform1f(glGetUniformLocation(p, "isoValue"), isoValue);
	glUniform1f(glGetUniformLocation(p, "alphaPower"), alphaPower);
	glUniform1f(glGetUniformLocation(p, "sliceDepth"), sliceDepth);

	glGenTextures(1, &transferFunctionTexture);
	glBindTexture(GL_TEXTURE_1D, transferFunctionTexture);

	// Upload transfer function data
	glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA, 256, 0, GL_RGBA, GL_FLOAT, transferFunction);

	// Set texture parameters for the transfer function
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);

}


int main(int argc, char** argv)
{
	glutInit(&argc, argv);

	//
	// 1. Transfer Function Editor Window
	//
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowPosition(100, 700);
	glutInitWindowSize(600, 300);
	transferFunctionWindow = glutCreateWindow("Transfer Function");

	// register callbacks
	glutDisplayFunc(renderScene_transferFunction);
	glutReshapeFunc(changeSize_transferFunction);

	glutMouseFunc(mouseClick_transferFunction);
	glutMotionFunc(mouseMove_transferFunction);
	glutIdleFunc(idle);

	init_transferFunction();

	//
	// 2. Main Volume Rendering Window
	//
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowPosition(100, 100);
	glutInitWindowSize(600, 600);
	volumeRenderingWindow = glutCreateWindow("Volume Rendering");

	// register callbacks
	glutDisplayFunc(renderScene);
	glutReshapeFunc(changeSize);
	glutKeyboardFunc(keyboard);
	glutSpecialFunc(specialKeys);

	glutMouseFunc(mouseClick);
	glutMotionFunc(mouseMove);
	glutMouseWheelFunc(mouseWheel);

	glutIdleFunc(idle);

	glEnable(GL_DEPTH_TEST);

	glewInit();
	if (glewIsSupported("GL_VERSION_3_3"))
		printf("Ready for OpenGL 3.3\n");
	else {
		printf("OpenGL 3.3 is not supported\n");
		exit(1);
	}

	// Create shader program
	p = createGLSLProgram("../volumeRendering.vert", NULL, "../volumeRendering.frag");

	init();

	// enter GLUT event processing cycle
	glutMainLoop();

	return 1;
}

mat4 rotate(float angle, const vec3& axis) {
	// Normalize the axis
	vec3 axis_normalized = normalize(axis);
	float cos_angle = cos(angle);
	float sin_angle = sin(angle);

	// Rotation matrix components
	float ux = axis_normalized.x;
	float uy = axis_normalized.y;
	float uz = axis_normalized.z;

	mat4 rotation_matrix;
	rotation_matrix[0][0] = cos_angle + ux * ux * (1 - cos_angle);
	rotation_matrix[0][1] = ux * uy * (1 - cos_angle) - uz * sin_angle;
	rotation_matrix[0][2] = ux * uz * (1 - cos_angle) + uy * sin_angle;
	rotation_matrix[1][0] = uy * ux * (1 - cos_angle) + uz * sin_angle;
	rotation_matrix[1][1] = cos_angle + uy * uy * (1 - cos_angle);
	rotation_matrix[1][2] = uy * uz * (1 - cos_angle) - ux * sin_angle;
	rotation_matrix[2][0] = uz * ux * (1 - cos_angle) - uy * sin_angle;
	rotation_matrix[2][1] = uz * uy * (1 - cos_angle) + ux * sin_angle;
	rotation_matrix[2][2] = cos_angle + uz * uz * (1 - cos_angle);

	// Return the rotation matrix
	return rotation_matrix;
}

