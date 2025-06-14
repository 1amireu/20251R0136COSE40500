/*
 * Skeleton code for COSE436 Fall 2024 Assignment 3
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
#include "Angel.h"

 //
 // Definitions
 //

typedef struct {
	unsigned char x, y, z, w;
} uchar4;
typedef unsigned char uchar;

typedef struct {
	GLuint VAO, VBO, EBO;
	GLuint texture;
	GLuint texWidth, texHeight; 
	GLfloat* vertices;
} GameObject;

typedef struct treenode {
	GameObject obj;
	mat4 m;                         
	vec3 jointOffset;
	void (*f)(const GameObject &obj, mat4 modelMatrix);
	struct treenode* sibling; 
	struct treenode* child; 
} treenode;

GameObject slime, grass, stone, oakLog, leaf;
struct treenode slimeNode;

GameObject steveHead, steveTorso, steveArmL, steveArmR, steveLegL, steveLegR;
struct treenode stHead, stTorso, stArmL, stArmR, stLegL, stLegR;

GameObject marioHead, marioTorso, marioArmL, marioArmR, marioLegL, marioLegR;
struct treenode mHead, mTorso, mArmL, mArmR, mLegL, mLegR;

GameObject grootHead, grootTorso, grootArmL, grootArmR, grootLegL, grootLegR;
struct treenode gHead, gTorso, gArmL, gArmR, gLegL, gLegR, gLeafL, gLeafR, gLeafH;

GameObject skybox, glossyCube;

// BMP loader
void LoadBMPFile(uchar4** dst, int* width, int* height, const char* name);

void mouseButton(int button, int state, int x, int y);
void mouseMotion(int x, int y);
void mouseWheel(int button, int dir, int x, int y);
void trackball_ptov(int x, int y, int width, int height, float v[3]);

void renderScenePlatform(int rows, int cols);
void renderTrees(mat4 modelMatrix);
void renderSkybox(const GameObject& obj, GLuint shaderProgram);
void renderGlossy(const GameObject& obj, GLuint shaderProgram, mat4 modelMatrix);
void drawObject(const GameObject& obj, mat4 modelMatrix);
void setupObjects();

void generateCuboid(GLfloat dims[3], const GLfloat textureCoords[6][4],
	GLfloat vertices[120], GLuint texWidth, GLuint texHeight);
void generateObject(GameObject& obj, GLfloat dims[3], const GLfloat textureCoords[6][4],
	GLuint texWidth, GLuint texHeight);
void generateSkybox(GameObject& obj);
void generateGlossyCube(GameObject& obj);

GLuint loadTexture(const char* textureFile);
GLuint loadCubemap(const char* cubemapFiles[6]);

mat4 rotate(float angle, const vec3& axis);
mat4 inverse(const mat4& m);

void traverse(treenode* node, mat4 currentModel);
void jointRotate(treenode* node, mat4 rotationMatrix);

// Shader programs
GLuint p, skyboxShader, glossyShader;

// Virtual Trackball
GLfloat lastPos[3] = { 0.0f, 0.0f, 1.0f };
bool leftMouse = false, rightMouse = false;
vec3 rotationAxis = vec3(0.0f);
GLfloat zoomFactor = 1.0f, dragFactor = 3.0f;

// Look-At matrix
vec4 eye = { 0.0f, 0.0f, 15.0f, 1.0f };
vec4 at = { 0.0f, 0.0f, 0.0f, 1.0f };
vec4 up = { 0.0f, 1.0f, 0.0f, 0.0f };
mat4 viewMatrix = LookAt(eye, at, up);
vec4 cameraPosition = eye;
vec4 upDir = up;

GLfloat scale = 1.0;
mat4 currentModel = mat4(1.0f);
mat4 cameraMatrix = mat4(1.0f);
mat4 projectionMatrix;
GLuint faceIndices[] = {
	0, 1, 2,  2, 3, 0, // Front face
	4, 5, 6,  6, 7, 4, // Back face
	8, 9, 10, 10, 11, 8, // Left face
	12, 13, 14, 14, 15, 12, // Right face
	16, 17, 18, 18, 19, 16, // Top face
	20, 21, 22, 22, 23, 20 // Bottom face
};

bool renderGrass = true, renderTree = true, renderMirror = false;;
void changeSize(int w, int h) {

	// Prevent a divide by zero, when window is too short
	// (you cant make a window of zero width).
	if (h == 0) h = 1;

	float ratio = 1.0f * (float)w / (float)h;

	// Set the viewport to be the entire window
	glViewport(0, 0, w, h);

}

void keyboard(unsigned char key, int x, int y)
{
	// ToDo
	if (key == 'g') {
		renderGrass = !renderGrass;
	}
	else if (key == 't') {
		renderTree = !renderTree;
	}
	else if (key == 'm') {
		renderMirror = !renderMirror;
	}

	glutPostRedisplay();
}

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
	v[2] *= a;
}

// Mouse click handler to start/stop dragging
void mouseButton(int button, int state, int x, int y) {
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
			rightMouse = true;  // Start translating
			lastPos[0] = x;
			lastPos[1] = y;
		}
		else if (state == GLUT_UP) {
			rightMouse = false;  // Stop translating
		}
	}
}

// Mouse motion handler
void mouseMotion(int x, int y) {
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

			mat4 rotationMatrix = transpose(rotate(angle * M_PI / 180, rotationAxis));
			cameraPosition = rotationMatrix * cameraPosition;
			upDir = rotationMatrix * upDir;
			viewMatrix = LookAt(cameraPosition, at, upDir);

		}
		glutPostRedisplay();
	}

	if (rightMouse) {
		/* Handle translation */
		dx = (float)(x - lastPos[0]) / winWidth * dragFactor;   // Normalize by window width
		dy = (float)(lastPos[1] - y) / winHeight * dragFactor; // Normalize by window height

		lastPos[0] = x;
		lastPos[1] = y;
		
		vec3 translationVector = vec3(dx, dy, 0.0f);
		mat4 inverseView = transpose(viewMatrix);
		vec4 transformedVec = inverseView * vec4(translationVector, 0.0f);

		currentModel = Translate(transformedVec.x, transformedVec.y, transformedVec.z) * currentModel;

		glutPostRedisplay();
	}
}

void mouseWheel(int button, int dir, int x, int y) {
	if (dir > 0) {
		zoomFactor *= 1.1f;  // Zoom in
	}
	else {
		zoomFactor /= 1.1f;  // Zoom out
	}

	// Ensure zoom stays within a reasonable range
	if (zoomFactor < 0.3f) zoomFactor = 0.3f;
	if (zoomFactor > 5.0f) zoomFactor = 5.0f;

	projectionMatrix = Perspective(45.0 / zoomFactor, 1.0, 0.1, 100.0);
	glutPostRedisplay(); // Trigger a redraw
}

void renderScene(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	renderSkybox(skybox, skyboxShader);

	glUseProgram(p);

	if (renderMirror) renderGlossy(glossyCube, glossyShader, currentModel);

	else {
		if (renderGrass) renderScenePlatform(18, 18);
		drawObject(slime, currentModel * Translate(0, -1, 0));
		traverse(&mTorso, currentModel);
		traverse(&stTorso, currentModel);
		traverse(&gTorso, currentModel);
		if (renderTree) {
			renderTrees(currentModel * Translate(-6, -1, -6));
			renderTrees(currentModel * Translate(-6, -1, 6));
			renderTrees(currentModel * Translate(6, -1, 6));
		}
		if (renderGrass) renderScenePlatform(18, 18);
	}

	glutSwapBuffers();
}

GLuint frame = 0, interval = 500;
GLuint storsoFull = 90, sheadFull = 20, sarmFull = 125, slegFull = 80;
double sincTorso = static_cast<double>(storsoFull) / interval;
double sincHead = static_cast<double>(sheadFull) / interval * 2;
double sincArm = static_cast<double>(sarmFull) / interval * 2;
double sincLeg = static_cast<double>(slegFull) / interval * 2;
bool grootGrow = true;
float grootGrowth = 1.0035;

void idle()
{	
	jointRotate(&stTorso, RotateY(sincTorso));
	if (grootGrow) gTorso.m *= Scale(vec3(grootGrowth)) * Translate(0, grootGrowth - 1, 0);
	else gTorso.m *= Scale(vec3(1/grootGrowth)) * Translate(0, -(grootGrowth - 1), 0);
	if (frame < interval / 2){
		jointRotate(&stHead, RotateX(-sincHead));
		jointRotate(&stArmL, RotateX(-sincArm));
		jointRotate(&stArmR, RotateX(sincArm));
		jointRotate(&stLegL, RotateX(sincLeg));
		jointRotate(&stLegR, RotateX(-sincLeg));
		mTorso.m *= Translate(0, 0.03, 0);

		if (frame < interval / 4) {
			jointRotate(&mHead, RotateX(-sincHead * 2));
			jointRotate(&mArmL, RotateX(-sincArm * 2));
			jointRotate(&mArmR, RotateX(sincArm * 2));
			jointRotate(&mLegL, RotateX(sincLeg * 2));
			jointRotate(&mLegR, RotateX(-sincLeg * 2));
		}
		else {
			jointRotate(&mHead, RotateX(sincHead * 2));
			jointRotate(&mArmL, RotateX(sincArm * 2));
			jointRotate(&mArmR, RotateX(-sincArm * 2));
			jointRotate(&mLegL, RotateX(-sincLeg * 2));
			jointRotate(&mLegR, RotateX(sincLeg * 2));
		}
	}
	else {
		jointRotate(&stHead, RotateX(sincHead));
		jointRotate(&stArmL, RotateX(sincArm));
		jointRotate(&stArmR, RotateX(-sincArm));
		jointRotate(&stLegL, RotateX(-sincLeg));
		jointRotate(&stLegR, RotateX(sincLeg));
		mTorso.m *= Translate(0, -0.03, 0);

		if (frame < interval / 4 * 3) {
			jointRotate(&mHead, RotateX(-sincHead * 2));
			jointRotate(&mArmL, RotateX(-sincArm * 2));
			jointRotate(&mArmR, RotateX(sincArm * 2));
			jointRotate(&mLegL, RotateX(sincLeg * 2));
			jointRotate(&mLegR, RotateX(-sincLeg * 2));
		}
		else {
			jointRotate(&mHead, RotateX(sincHead * 2));
			jointRotate(&mArmL, RotateX(sincArm * 2));
			jointRotate(&mArmR, RotateX(-sincArm * 2));
			jointRotate(&mLegL, RotateX(-sincLeg * 2));
			jointRotate(&mLegR, RotateX(sincLeg * 2));
		}
	}
	frame++;
	if (frame == interval) {
		frame = 0;
		grootGrow = !grootGrow;
	}
	glutPostRedisplay();
}

void init()
{
	glEnable(GL_DEPTH_TEST);
	glClearColor(1.0, 1.0, 1.0, 1.0);

	// Create shader program
	p = createGLSLProgram("../vshader.vert", NULL, "../fshader.frag");
	skyboxShader = createGLSLProgram("../vskybox.vert", NULL, "../fskybox.frag");
	glossyShader = createGLSLProgram("../vglossy.vert", NULL, "../fglossy.frag");

	glEnable(GL_DEPTH_TEST);
	glClearColor(1.0, 1.0, 1.0, 1.0);

	setupObjects();
	projectionMatrix = Perspective(45.0 / zoomFactor, 1.0, 0.1, 100.0);
}

int main(int argc, char** argv) {

	// init GLUT and create Window
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowPosition(400, 250);
	glutInitWindowSize(600, 600);
	glutCreateWindow("COSE436 - Assignment 3");

	// register callbacks
	glutDisplayFunc(renderScene);
	glutIdleFunc(renderScene);
	glutReshapeFunc(changeSize);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouseButton);
	glutMotionFunc(mouseMotion);
	glutMouseWheelFunc(mouseWheel);
	glutIdleFunc(idle);

	glewInit();
	if (glewIsSupported("GL_VERSION_3_3"))
		printf("Ready for OpenGL 3.3\n");
	else {
		printf("OpenGL 3.3 is not supported\n");
		exit(1);
	}
	init();
	// Entering GLUT event processing cycle
	glutMainLoop();

	return 1;
}

void renderScenePlatform(int rows, int cols) {
	// Define the center of the grid and the circle radius
	float centerX = rows / 2.0f;
	float centerZ = cols / 2.0f;
	float circleRadius = std::min(rows, cols) / 4.0f;  // Adjust the radius as needed

	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			// Calculate position
			float x = i - centerX;
			float z = j - centerZ;

			// Compute the distance from the center
			float distanceFromCenter = sqrt(x * x + z * z);

			// Select the object to draw based on the distance
			if (distanceFromCenter < circleRadius) {
				// Draw stone if inside the circle
				mat4 model = currentModel * Translate(vec3(x, -2.0f, z));
				drawObject(stone, model);
			}
			else {
				// Draw grass otherwise
				mat4 model = currentModel * Translate(vec3(x, -2.0f, z));
				drawObject(grass, model);
			}
		}
	}
}

void renderTrees(mat4 modelMatrix) {
	// Parameters for the tree
	int trunkHeight = 3;          // Number of log blocks in the trunk
	int leafRadius = 2;           // Horizontal radius of the leaf canopy
	int leafHeight = 3;           // Height of the leaf canopy

	// Render the trunk
	for (int i = 0; i < trunkHeight; ++i) {
		// Offset each log block upwards by 1 unit (block height)
		mat4 trunkModel = modelMatrix * Translate(0.0f, i, 0.0f);
		drawObject(oakLog, trunkModel);
	}

	// Render the leaves
	for (int y = 0; y < leafHeight; ++y) {               // Loop through leaf layers
		for (int x = -leafRadius; x <= leafRadius; ++x) { // Loop through x-axis
			for (int z = -leafRadius; z <= leafRadius; ++z) { // Loop through z-axis
				// Skip blocks outside the circular radius
				if (sqrt(x * x + z * z) > leafRadius - y) continue;

				// Position the leaf block above the trunk
				mat4 leafModel = modelMatrix * Translate(x, trunkHeight + y, z);
				drawObject(leaf, leafModel);
			}
		}
	}
}

void renderSkybox(const GameObject& obj, GLuint shaderProgram) {

	glDepthMask(GL_FALSE); // Disable depth writing
	glUseProgram(shaderProgram);

	glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "viewMatrix"), 1, GL_TRUE, viewMatrix);
	glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projectionMatrix"), 1, GL_TRUE, projectionMatrix);

	glUniform1i(glGetUniformLocation(shaderProgram, "skybox"), 0);

	// Bind the cubemap texture
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_CUBE_MAP, obj.texture);

	// Draw the skybox cube
	glBindVertexArray(obj.VAO);
	glDrawArrays(GL_TRIANGLES, 0, 36);
	glBindVertexArray(0);

	glDepthMask(GL_TRUE); // Re-enable depth writing
}

void renderGlossy(const GameObject& obj, GLuint shaderProgram, mat4 modelMatrix) {

	glUseProgram(shaderProgram);

	glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "modelMatrix"), 1, GL_TRUE, modelMatrix);
	glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "viewMatrix"), 1, GL_TRUE, viewMatrix);
	glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projectionMatrix"), 1, GL_TRUE, projectionMatrix);

	glUniform3fv(glGetUniformLocation(shaderProgram, "cameraPos"), 1, cameraPosition);
	glUniform1i(glGetUniformLocation(shaderProgram, "skybox"), 0);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_CUBE_MAP, obj.texture);

	glBindVertexArray(obj.VAO);
	glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
	glBindVertexArray(0);
}

void drawObject(const GameObject& obj, mat4 modelMatrix) {

	glUniformMatrix4fv(glGetUniformLocation(p, "modelMatrix"), 1, GL_TRUE, modelMatrix);
	glUniformMatrix4fv(glGetUniformLocation(p, "viewMatrix"), 1, GL_TRUE, viewMatrix);
	glUniformMatrix4fv(glGetUniformLocation(p, "projectionMatrix"), 1, GL_TRUE, projectionMatrix);

	// Set the texture uniform
	glUniform1i(glGetUniformLocation(p, "texture"), 0); // Use texture unit GL_TEXTURE0

	// Activate and bind the object's texture
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, obj.texture);

	// Bind the object's VAO and draw
	glBindVertexArray(obj.VAO);
	glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
	glBindVertexArray(0);

}

void setupObjects() {
	//Define texture coordinates for different objects
	GLfloat slimeTexCoords[6][4] = {
		{193, 575, 384 ,384}, // Front
		{193, 191, 384, 1}, // Back
		{1, 383, 192, 192}, // Left
		{385, 383, 576, 192}, // Right
		{192, 383, 384, 192}, // Top
		{193, 767, 384, 624}  // Bottom
	};
	GLfloat grassTexCoords[6][4] = {
		{610, 607, 907, 307}, // Front
		{610, 607, 907, 307}, // Back
		{610, 607, 907, 307}, // Left
		{610, 607, 907, 307}, // Right
		{610, 303, 907, 3}, // Top
		{610, 911, 907, 611}  // Bottom
	};
	GLfloat stoneTexCoords[6][4] = {
		{610, 607, 907, 307},
		{610, 607, 907, 307},
		{610, 607, 907, 307},
		{610, 607, 907, 307},
		{610, 607, 907, 307},
		{610, 607, 907, 307}
	};
	GLfloat logTexCoords[6][4] = {
		{0, 32, 16, 17},
		{0, 32, 16, 17},
		{0, 32, 16, 17},
		{0, 32, 16, 17},
		{0, 16, 16, 0}
	};
	GLfloat leafTexCoords[6][4] = {
		{0, 800, 800, 0},
		{0, 800, 800, 0},
		{0, 800, 800, 0},
		{0, 800, 800, 0},
		{0, 800, 800, 0},
		{0, 800, 800, 0}
	};

	GLfloat steveTorsoTexCoords[6][4] = {
		{450, 924, 259, 1115}, // Front
		{839, 924, 649, 1115}, // Back
		{645, 924, 454, 1167}, // Left
		{255, 924, 94, 1167}, // Right
		{450, 1094, 259, 1115}, // Top
		{450, 729, 259, 920}  // Bottom
	};
	GLfloat steveHeadTexCoords[6][4] = {
		{818, 580, 1117, 373}, // Front
		{211, 580, 510, 373}, // Back
		{515, 580, 814, 373}, // Left
		{1122, 580, 1421, 373}, // Right
		{818, 368, 1116, 69}, // Top
		{818, 884, 1117, 585}  // Bottom
	};
	GLfloat steveArmLTexCoords[6][4] = {
		{1199, 1038, 1269, 902}, // Front
		{1348, 1038, 1418, 902}, // Back
		{1199, 1038, 1215, 902}, // Left
		{1274, 1038, 1344, 902}, // Right
		{1199, 898, 1269, 828}, // Top
		{1344, 1113, 1273, 1042}  // Bottom
	};
	GLfloat steveArmRTexCoords[6][4] = {
		{245, 248, 314, 113}, // Front
		{96, 248, 165, 118}, // Back
		{170, 248, 240, 113}, // Left
		{1199, 1038, 1215, 902}, // Right
		{1199, 898, 1269, 828}, // Top
		{1344, 1113, 1273, 1042}  // Bottom
	};
	GLfloat steveLegLTexCoords[6][4] = {
		{574, 1471, 641, 1402}, // Front
		{283, 1471, 349, 1402}, // Back
		{429, 1471, 569, 1371}, // Left
		{720, 1471, 860, 1371}, // Right
		{647, 1472, 715, 1472}, // Top
		{428, 1472, 570, 1472}  // Bottom
	};
	GLfloat steveLegRTexCoords[6][4] = {
		{574, 1471, 641, 1402}, // Front
		{283, 1471, 349, 1402}, // Back
		{720, 1471, 860, 1371}, // Left
		{429, 1471, 569, 1371}, // Right
		{647, 1472, 715, 1472}, // Top
		{428, 1472, 570, 1472}  // Bottom
	};

	GLfloat pmskinTorsoTexCoords[6][4] = {
		{548, 187, 640, 47},
		{640, 235, 548, 375},
		{501, 187, 547, 47},
		{641, 187, 688, 47},
		{548, 46, 640, 1},
		{548, 234, 640, 188}
	};
	GLfloat pmskinHeadTexCoords[6][4] = {
		{102, 709, 194, 617},
		{194, 429, 103, 521},
		{8, 709, 101, 617},
		{195, 709, 288, 617},
		{102, 616, 194, 522},
		{194, 710, 102, 803}
	};
	GLfloat pmskinArmLTexCoords[6][4] = {
		{595, 879, 640, 739},
		{513, 879, 547, 739},
		{548, 879, 594, 739},
		{641, 879, 676, 739},
		{548, 738, 582, 692},
		{548, 926, 582, 880}
	};
	GLfloat pmskinArmRTexCoords[6][4] = {
		{353, 879, 387, 739},
		{434, 879, 468, 739},
		{306, 879, 352, 739},
		{388, 879, 433, 739},
		{388, 738, 422, 694},
		{388, 926, 422, 880}
	};
	GLfloat pmskinLegLTexCoords[6][4] = {
		{548, 596, 594, 456},
		{641, 596, 688, 456},
		{501, 596, 547, 456},
		{595, 596, 640, 456},
		{548, 455, 594, 409},
		{548, 642, 594, 597}
	};
	GLfloat pmskinLegRTexCoords[6][4] = {
		{400, 596, 445, 456},
		{306, 596, 352, 456},
		{353, 596, 399, 456},
		{446, 596, 492, 456},
		{400, 455, 445, 409},
		{400, 642, 445, 597}
	};

	GLfloat cube[3] = { 1, 1, 1 };
	GLfloat torso[3] = { 4, 6, 2 };
	GLfloat head[3] = { 4, 4, 4 };
	GLfloat limbs[3] = { 2, 6, 2 };

	float scale = 0.125;
	for (int i = 0; i < 3; i++) {
		torso[i] *= scale;
		head[i] *= scale;
		limbs[i] *= scale;
	}

	// Generate the objects
	generateObject(slime, cube, slimeTexCoords, 578, 769);
	generateObject(grass, cube, grassTexCoords, 1214, 914);
	generateObject(stone, cube, stoneTexCoords, 1214, 914);
	generateObject(oakLog, cube, logTexCoords, 16, 32);
	generateObject(leaf, cube, leafTexCoords, 800, 800);

	/* Steve object generation */ {
		generateObject(steveTorso, torso, steveTorsoTexCoords, 1482, 1173);
		generateObject(steveHead, head, steveHeadTexCoords, 1482, 1173);
		generateObject(steveLegL, limbs, steveLegLTexCoords, 1173, 1482);
		generateObject(steveLegR, limbs, steveLegRTexCoords, 1173, 1482);
		generateObject(steveArmL, limbs, steveArmLTexCoords, 1482, 1173);
		generateObject(steveArmR, limbs, steveArmRTexCoords, 1482, 1173);
	}
	/* Mario object generation */ {
		generateObject(marioTorso, torso, pmskinTorsoTexCoords, 710, 928);
		generateObject(marioHead, head, pmskinHeadTexCoords, 710, 928);
		generateObject(marioLegL, limbs, pmskinLegLTexCoords, 710, 928);
		generateObject(marioLegR, limbs, pmskinLegRTexCoords, 710, 928);
		generateObject(marioArmL, limbs, pmskinArmLTexCoords, 710, 928);
		generateObject(marioArmR, limbs, pmskinArmRTexCoords, 710, 928);
	}
	/* Groot object generation */ {
		generateObject(grootTorso, torso, pmskinTorsoTexCoords, 710, 928);
		generateObject(grootHead, head, pmskinHeadTexCoords, 710, 928);
		generateObject(grootLegL, limbs, pmskinLegLTexCoords, 710, 928);
		generateObject(grootLegR, limbs, pmskinLegRTexCoords, 710, 928);
		generateObject(grootArmL, limbs, pmskinArmLTexCoords, 710, 928);
		generateObject(grootArmR, limbs, pmskinArmRTexCoords, 710, 928);
	}
	// Load textures
	slime.texture = loadTexture("../mob.bmp");
	grass.texture = loadTexture("../grass.bmp");
	stone.texture = loadTexture("../stone.bmp");
	oakLog.texture = loadTexture("../oak_log.bmp");
	leaf.texture = loadTexture("../leaf.bmp");
	
	mat4 model;

	/* Steve texture binding */ {
		GLuint steveTex = loadTexture("../steve.bmp");
		GLuint steveLegTex = loadTexture("../steveleg.bmp");
		steveHead.texture = steveTex;
		steveTorso.texture = steveTex;
		steveArmL.texture = steveTex;
		steveArmR.texture = steveTex;
		steveLegL.texture = steveLegTex;
		steveLegR.texture = steveLegTex;
	}
	/* Steve treenode intialization */ {
		stTorso = { steveTorso, Translate(-5, -0.325, 0), vec3(5, 0, 0), drawObject, NULL, &stHead };
		model = Translate(0, -.0625, 0) * RotateX(10) * Translate(0, .0625, 0);
		stHead = { steveHead, Translate(0.0f, .625f, 0.0f) * model, vec3(0.0f, -.0625f, 0.0f), drawObject, &stLegL, NULL };
		model = Translate(0, .325, 0) * RotateX(-40) * Translate(0, -.325, 0);
		stLegL = { steveLegL, Translate(.125f, -.75f, 0.0f) * model, vec3(0.0f, .325f, 0.0f), drawObject, &stLegR, NULL };
		model = Translate(0, .325, 0) * RotateX(40) * Translate(0, -.325, 0);
		stLegR = { steveLegR, Translate(-.125f, -.75f, 0.0f) * model, vec3(0.0f, .325f, 0.0f), drawObject, &stArmL, NULL };
		model = Translate(0, .25, 0) * RotateX(50) * Translate(0, -.25, 0);
		stArmL = { steveArmL, Translate(.375f, 0.0f, 0.0f) * model, vec3(0.0f, .25f, 0.0f), drawObject, &stArmR, NULL };
		model = Translate(0, .25, 0) * RotateX(50 - 125) * Translate(0, -.25, 0);
		stArmR = { steveArmR, Translate(-.375f, 0.0f, 0.0f) * model, vec3(0.0f, .25f, 0.0f), drawObject, NULL, NULL };
	}
	/* Mario texture binding */ {
		GLuint marioTex = loadTexture("../marioskin.bmp");
		marioHead.texture = marioTex;
		marioTorso.texture = marioTex;
		marioArmL.texture = marioTex;
		marioArmR.texture = marioTex;
		marioLegL.texture = marioTex;
		marioLegR.texture = marioTex;
	}
	/* Mario treenode initialization */ {
		mTorso = { marioTorso, Translate(0, .5, 0), vec3(0, 0, 0), drawObject, NULL, &mHead };
		model = Translate(0, -.0625, 0) * RotateX(10) * Translate(0, .0625, 0);
		mHead = { marioHead, Translate(0.0f, .625f, 0.0f) * model, vec3(0.0f, -.0625f, 0.0f), drawObject, &mLegL, NULL };
		model = Translate(0, .325, 0) * RotateX(-40) * Translate(0, -.325, 0);
		mLegL = { marioLegL, Translate(.125f, -.75f, 0.0f) * model, vec3(0.0f, .325f, 0.0f), drawObject, &mLegR, NULL };
		model = Translate(0, .325, 0) * RotateX(40) * Translate(0, -.325, 0);
		mLegR = { marioLegR, Translate(-.125f, -.75f, 0.0f) * model, vec3(0.0f, .325f, 0.0f), drawObject, &mArmL, NULL };
		model = Translate(0, .25, 0) * RotateX(50) * Translate(0, -.25, 0);
		mArmL = { marioArmL, Translate(.375f, 0.0f, 0.0f) * model, vec3(0.0f, .25f, 0.0f), drawObject, &mArmR, NULL };
		model = Translate(0, .25, 0) * RotateX(50 - 125) * Translate(0, -.25, 0);
		mArmR = { marioArmR, Translate(-.375f, 0.0f, 0.0f) * model, vec3(0.0f, .25f, 0.0f), drawObject, NULL, NULL };
	}
	/* Groot texture binding */ {
		GLuint grootTex = loadTexture("../grootskin.bmp");
		grootHead.texture = grootTex;
		grootTorso.texture = grootTex;
		grootArmL.texture = grootTex;
		grootArmR.texture = grootTex;
		grootLegL.texture = grootTex;
		grootLegR.texture = grootTex;
	}
	/* Groot treenode initialization */ {
		gTorso = { grootTorso, Translate(6, -.325, -6) * RotateY(-30), vec3(0, 0, 0), drawObject, NULL, &gHead};
		model = Translate(0, -.0625, 0) * RotateX(-10) * Translate(0, .0625, 0);
		gHead = { grootHead, Translate(0.0f, .625f, 0.0f) * model, vec3(0.0f, -.0625f, 0.0f), drawObject, &gLegL, &gLeafH };
		model = Translate(0, .325, 0) * RotateZ(15) * Translate(0, -.325, 0);
		gLegL = { grootLegL, Translate(.125f, -.75f, 0.0f) * model, vec3(0.0f, .325f, 0.0f), drawObject, &gLegR, NULL };
		model = Translate(0, .325, 0) * RotateZ(-15) * Translate(0, -.325, 0);
		gLegR = { grootLegR, Translate(-.125f, -.75f, 0.0f) * model, vec3(0.0f, .325f, 0.0f), drawObject, &gArmL, NULL };
		model = Translate(0, .25, 0) * RotateZ(140) * Translate(0, -.25, 0);
		gArmL = { grootArmL, Translate(.375f, 0.0f, 0.0f) * model, vec3(0.0f, .25f, 0.0f), drawObject, &gArmR, &gLeafL };
		model = Translate(0, .25, 0) * RotateZ(-140) * Translate(0, -.25, 0);
		gArmR = { grootArmR, Translate(-.375f, 0.0f, 0.0f) * model, vec3(0.0f, .25f, 0.0f), drawObject, NULL, &gLeafR };
		gLeafL = { leaf, Translate(0, -.625, 0) * Scale(vec3(.75)), vec3(0.0f), drawObject, NULL, NULL};
		gLeafR = { leaf, Translate(0, -.625, 0) * Scale(vec3(.75)), vec3(0.0f), drawObject, NULL, NULL };
		gLeafH = { leaf, Translate(0, .5, 0) * Scale(vec3(.75)), vec3(0.0f), drawObject, NULL, NULL };
	}

	generateSkybox(skybox);

	// Load Cubemap
	const char* cubemapFiles[6] = {
		"../right.bmp",
		"../left.bmp",
		"../top.bmp",
		"../bottom.bmp",
		"../front.bmp",
		"../back.bmp"
	};
	
	GLuint skyboxTex = loadCubemap(cubemapFiles);
	skybox.texture = skyboxTex;

	generateGlossyCube(glossyCube);
	glossyCube.texture = skyboxTex;
}

// A utility function to generate the cube
void generateCuboid(GLfloat dims[3],
	const GLfloat textureCoords[6][4],
	GLfloat vertices[120], GLuint texWidth, GLuint texHeight) {

	GLfloat hD[3] = { dims[0] / 2.0f, dims[1] / 2.0f, dims[2] / 2.0f };

	// Define the cube's vertices for each face (6 faces, 4 vertices per face)
	GLfloat cubeVertices[] = {
		// Front face
		-hD[0], -hD[1],  hD[2],
		 hD[0], -hD[1],  hD[2],
		 hD[0],  hD[1],  hD[2],
		-hD[0],  hD[1],  hD[2],

		// Back face
		 hD[0], -hD[1], -hD[2],
		-hD[0], -hD[1], -hD[2],
		-hD[0],  hD[1], -hD[2],
		 hD[0],  hD[1], -hD[2],

		// Left face
		-hD[0], -hD[1], -hD[2],
		-hD[0], -hD[1],  hD[2],
		-hD[0],  hD[1],  hD[2],
		-hD[0],  hD[1], -hD[2],

		// Right face
		 hD[0], -hD[1],  hD[2],
		 hD[0], -hD[1], -hD[2],
		 hD[0],  hD[1], -hD[2],
		 hD[0],  hD[1],  hD[2],

		 // Top face
		-hD[0],  hD[1],  hD[2],
		 hD[0],  hD[1],  hD[2],
		 hD[0],  hD[1], -hD[2],
		-hD[0],  hD[1], -hD[2],

		// Bottom face
	   -hD[0], -hD[1], -hD[2],
		hD[0], -hD[1], -hD[2],
		hD[0], -hD[1],  hD[2],
	   -hD[0], -hD[1],  hD[2],
	};

	// Initialize the output array to store positions and texture coordinates
	int vertexIndex = 0;

	// Loop through the faces and combine positions and texture coordinates
	for (int i = 0; i < 6; ++i) {
		GLfloat u0 = textureCoords[i][0] / texWidth;  // Bottom-left u
		GLfloat v0 = (texHeight - textureCoords[i][1]) / texHeight;  // Bottom-left v
		GLfloat u1 = textureCoords[i][2] / texWidth;  // Top-right u
		GLfloat v1 = (texHeight - textureCoords[i][3]) / texHeight;  // Top-right v

		// For each face, add the 4 vertices with both position and texture coordinates
		for (int j = 0; j < 4; ++j) {
			// Add position coordinates
			vertices[vertexIndex++] = cubeVertices[12 * i + 0 + 3 * j];    // x
			vertices[vertexIndex++] = cubeVertices[12 * i + 1 + 3 * j];    // y
			vertices[vertexIndex++] = cubeVertices[12 * i + 2 + 3 * j];    // z

			// Add calculated texture coordinates
			if (j == 0) {
				// Bottom-left
				vertices[vertexIndex++] = u0;
				vertices[vertexIndex++] = v0;
			}
			else if (j == 1) {
				// Bottom-right
				vertices[vertexIndex++] = u1;
				vertices[vertexIndex++] = v0;
			}
			else if (j == 2) {
				// Top-right
				vertices[vertexIndex++] = u1;
				vertices[vertexIndex++] = v1;
			}
			else {
				// Top-left
				vertices[vertexIndex++] = u0;
				vertices[vertexIndex++] = v1;
			}
		}
	}
}

void generateObject(GameObject& obj, GLfloat dims[3], const GLfloat textureCoords[6][4],
	GLuint texWidth, GLuint texHeight) {
	obj.vertices = new GLfloat[120];  // 6 faces * 2 triangles * 3 vertices per triangle
	obj.texWidth = texWidth;
	obj.texHeight = texHeight;

	generateCuboid(dims, textureCoords, obj.vertices, obj.texWidth, obj.texHeight);

	// Create new VAO, VBO, EBO
	glGenVertexArrays(1, &obj.VAO);
	glGenBuffers(1, &obj.VBO);
	glGenBuffers(1, &obj.EBO);

	glBindVertexArray(obj.VAO);

	glBindBuffer(GL_ARRAY_BUFFER, obj.VBO);
	glBufferData(GL_ARRAY_BUFFER, 120 * sizeof(GLfloat), obj.vertices, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, obj.EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(faceIndices), faceIndices, GL_STATIC_DRAW);

	// Position attribute
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (void*)0);
	glEnableVertexAttribArray(0);

	// Texture coordinate attribute
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (void*)(3 * sizeof(GLfloat)));
	glEnableVertexAttribArray(1);

	glBindVertexArray(0);
}

void generateSkybox(GameObject& obj) {
	// Skybox vertices: positions (each vertex position doubles as the texture coordinate)
	GLfloat skyboxVertices[] = {
		// positions          
		-1.0f,  1.0f, -1.0f,
		-1.0f, -1.0f, -1.0f,
		 1.0f, -1.0f, -1.0f,
		 1.0f, -1.0f, -1.0f,
		 1.0f,  1.0f, -1.0f,
		-1.0f,  1.0f, -1.0f,

		-1.0f, -1.0f,  1.0f,
		-1.0f, -1.0f, -1.0f,
		-1.0f,  1.0f, -1.0f,
		-1.0f,  1.0f, -1.0f,
		-1.0f,  1.0f,  1.0f,
		-1.0f, -1.0f,  1.0f,

		 1.0f, -1.0f, -1.0f,
		 1.0f, -1.0f,  1.0f,
		 1.0f,  1.0f,  1.0f,
		 1.0f,  1.0f,  1.0f,
		 1.0f,  1.0f, -1.0f,
		 1.0f, -1.0f, -1.0f,

		-1.0f, -1.0f,  1.0f,
		-1.0f,  1.0f,  1.0f,
		 1.0f,  1.0f,  1.0f,
		 1.0f,  1.0f,  1.0f,
		 1.0f, -1.0f,  1.0f,
		-1.0f, -1.0f,  1.0f,

		-1.0f,  1.0f, -1.0f,
		 1.0f,  1.0f, -1.0f,
		 1.0f,  1.0f,  1.0f,
		 1.0f,  1.0f,  1.0f,
		-1.0f,  1.0f,  1.0f,
		-1.0f,  1.0f, -1.0f,

		-1.0f, -1.0f, -1.0f,
		-1.0f, -1.0f,  1.0f,
		 1.0f, -1.0f, -1.0f,
		 1.0f, -1.0f, -1.0f,
		-1.0f, -1.0f,  1.0f,
		 1.0f, -1.0f,  1.0f
	};

	// Create VAO and VBO
	glGenVertexArrays(1, &obj.VAO);
	glGenBuffers(1, &obj.VBO);

	glBindVertexArray(obj.VAO);

	glBindBuffer(GL_ARRAY_BUFFER, obj.VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(skyboxVertices), skyboxVertices, GL_STATIC_DRAW);

	// Position attribute
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (void*)0);
	

	glBindVertexArray(0);
}

void generateGlossyCube(GameObject& obj) {
	GLfloat cubeVertices[] = {
		// positions          // normals
		-2.0f, -2.0f,  2.0f,  0.0f,  0.0f,  1.0f,
		 2.0f, -2.0f,  2.0f,  0.0f,  0.0f,  1.0f,
		 2.0f,  2.0f,  2.0f,  0.0f,  0.0f,  1.0f,
		-2.0f,  2.0f,  2.0f,  0.0f,  0.0f,  1.0f,

		-2.0f, -2.0f, -2.0f,  0.0f,  0.0f, -1.0f,
		 2.0f, -2.0f, -2.0f,  0.0f,  0.0f, -1.0f,
		 2.0f,  2.0f, -2.0f,  0.0f,  0.0f, -1.0f,
		-2.0f,  2.0f, -2.0f,  0.0f,  0.0f, -1.0f,

		-2.0f, -2.0f, -2.0f, -1.0f,  0.0f,  0.0f,
		-2.0f, -2.0f,  2.0f, -1.0f,  0.0f,  0.0f,
		-2.0f,  2.0f,  2.0f, -1.0f,  0.0f,  0.0f,
		-2.0f,  2.0f, -2.0f, -1.0f,  0.0f,  0.0f,

		 2.0f, -2.0f, -2.0f,  1.0f,  0.0f,  0.0f,
		 2.0f, -2.0f,  2.0f,  1.0f,  0.0f,  0.0f,
		 2.0f,  2.0f,  2.0f,  1.0f,  0.0f,  0.0f,
		 2.0f,  2.0f, -2.0f,  1.0f,  0.0f,  0.0f,

		-2.0f,  2.0f, -2.0f,  0.0f,  1.0f,  0.0f,
		 2.0f,  2.0f, -2.0f,  0.0f,  1.0f,  0.0f,
		 2.0f,  2.0f,  2.0f,  0.0f,  1.0f,  0.0f,
		-2.0f,  2.0f,  2.0f,  0.0f,  1.0f,  0.0f,

		-2.0f, -2.0f, -2.0f,  0.0f, -1.0f,  0.0f,
		 2.0f, -2.0f, -2.0f,  0.0f, -1.0f,  0.0f,
		 2.0f, -2.0f,  2.0f,  0.0f, -1.0f,  0.0f,
		-2.0f, -2.0f,  2.0f,  0.0f, -1.0f,  0.0f,
	};

	// Create new VAO, VBO, EBO
	glGenVertexArrays(1, &obj.VAO);
	glGenBuffers(1, &obj.VBO);
	glGenBuffers(1, &obj.EBO);

	glBindVertexArray(obj.VAO);

	glBindBuffer(GL_ARRAY_BUFFER, obj.VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(cubeVertices), cubeVertices, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, obj.EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(faceIndices), faceIndices, GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (void*)0);
	
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (void*)(3 * sizeof(GLfloat)));
	
	glBindVertexArray(0);
}

GLuint loadTexture(const char* textureFile) {
    int width, height;
    uchar4* dst;
    LoadBMPFile(&dst, &width, &height, textureFile);

    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, dst);
	free(dst);

    return texture;
}

GLuint loadCubemap(const char* cubemapFiles[6]) {

	GLuint texture;
	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_CUBE_MAP, texture);

	int width, height;
	uchar4* dst;

	for (GLuint i = 0; i < 6; ++i) {
		LoadBMPFile(&dst, &width, &height, cubemapFiles[i]);
		if (dst) {
			// Load each face of the cubemap
			glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGBA,
				width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, dst);
			free(dst);
		}
		else {
			std::cerr << "Failed to load texture for cubemap face: " << cubemapFiles[i] << std::endl;
		}
	}
	
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

	return texture;
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

void traverse(treenode* node, mat4 currentModel) {
	if (node == NULL) return;

	// Update the model matrix for this node
	mat4 newModel = currentModel * node->m;

	// Perform the node-specific operation using the updated matrix
	if (node->f) {
		node->f(node->obj, newModel);
	}

	// Traverse child with the updated matrix
	traverse(node->child, newModel);

	// Traverse sibling with the current matrix unchanged
	traverse(node->sibling, currentModel);
}

void jointRotate(treenode* node, mat4 rotationMatrix) {
	node->m *= Translate(node->jointOffset) * rotationMatrix * Translate(-node->jointOffset);

}

mat4 inverse(const mat4& m) {
	mat4 inv;

	// Transpose the rotation part
	for (int row = 0; row < 3; ++row) {
		for (int col = 0; col < 3; ++col) {
			inv[row][col] = m[col][row];
		}
	}

	float T0 = m[0][3];
	float T1 = m[1][3];
	float T2 = m[2][3];

	// Compute -R^T * T
	inv[0][3] = -(inv[0][0] * T0 + inv[0][1] * T1 + inv[0][2] * T2);
	inv[1][3] = -(inv[1][0] * T0 + inv[1][1] * T1 + inv[1][2] * T2);
	inv[2][3] = -(inv[2][0] * T0 + inv[2][1] * T1 + inv[2][2] * T2);

	inv[3][0] = 0.0f;
	inv[3][1] = 0.0f;
	inv[3][2] = 0.0f;
	inv[3][3] = 1.0f;

	return inv;
}




