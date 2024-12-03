// Thomas AUbourg & Fantin Charles


struct Ray{
    vec3 o;// Origin
    vec3 d;// Direction
};

struct Val {
  float v; // Signed distance
  int c; // Cost
};

// Compute point on ray
// ray : The ray
//   t : Distance
vec3 Point(Ray ray,float t)
{
    return ray.o+t*ray.d;
}

// Random direction in a hemisphere
// seed : Integer seed, from 0 to N
//    n : Direction of the hemisphere
vec3 Hemisphere(int seed,vec3 n)
{
    float a=fract(sin(176.19*float(seed)));// Uniform randoms
    float b=fract(sin(164.19*float(seed)));
    
    float u=2.*3.1415*a;// Random angle
    float v=acos(2.*b-1.);// Arccosine distribution to compensate at poles
    
    vec3 d=vec3(cos(u)*cos(v),sin(u)*cos(v),sin(v));// Direction
    if(dot(d,n)<0.){d=-d;}// Hemisphere
    
    return d;
}

// Camera -------------------------------------------------------------------------------

// Rotation matrix around z axis
// a : Angle
mat3 Rz(float a)
{
  float sa=sin(a);float ca=cos(a);
  return mat3(ca,sa,0.,-sa,ca,0.,0.,0.,1.);
}

// Compute the ray
//      m : Mouse position
//      p : Pixel
Ray CreateRay(vec2 m,vec2 p)
{
  float a=3.*3.14*m.x;
  float le=3.5;
  
  // Origin
  vec3 ro=vec3(37.,0.,15.);
  ro*=Rz(a);
  
  // Target point
  vec3 ta=vec3(0.,0.,1.);
  
  // Orthonormal frame
  vec3 w=normalize(ta-ro);
  vec3 u=normalize(cross(w,vec3(0.,0.,1.)));
  vec3 v=normalize(cross(u,w));
  vec3 rd=normalize(p.x*u+p.y*v+le*w);
  return Ray(ro,rd);
}

// Noise -------------------------------------------------------------------------------

// Gradient function for 3D
vec3 g(ivec3 q) {
    int n = q.x + q.y * 11111 + q.z * 33333;
    n = (n << 13) ^ n;
    n = (n * (n * n * 15731 + 789221) + 1376312589) >> 16;
    float angle1 = float(n);
    float angle2 = float(n * 57);
    return vec3(cos(angle1) * cos(angle2), sin(angle1) * cos(angle2), sin(angle2));
}

// 3D noise function
float noise(vec3 p) {
    ivec3 q = ivec3(floor(p)); // Grid cell coordinates
    vec3 f = fract(p); // Fractional part of the position within the cell
    vec3 u = f * f * (3.0 - 2.0 * f); // Smootherstep interpolation function

    // Trilinear interpolation of the noise values
    return mix(
        mix(
            mix(dot(g(q + ivec3(0, 0, 0)), f - vec3(0.0, 0.0, 0.0)),
                dot(g(q + ivec3(1, 0, 0)), f - vec3(1.0, 0.0, 0.0)), u.x),
            mix(dot(g(q + ivec3(0, 1, 0)), f - vec3(0.0, 1.0, 0.0)),
                dot(g(q + ivec3(1, 1, 0)), f - vec3(1.0, 1.0, 0.0)), u.x), u.y),
        mix(
            mix(dot(g(q + ivec3(0, 0, 1)), f - vec3(0.0, 0.0, 1.0)),
                dot(g(q + ivec3(1, 0, 1)), f - vec3(1.0, 0.0, 1.0)), u.x),
            mix(dot(g(q + ivec3(0, 1, 1)), f - vec3(0.0, 1.0, 1.0)),
                dot(g(q + ivec3(1, 1, 1)), f - vec3(1.0, 1.0, 1.0)), u.x), u.y), u.z);
}



// Fractional Brownian Motion (fBM) function
// p: position in space
// lacunarity: frequency factor between octaves
// gain: amplitude factor between octaves
// octaves: number of octaves to use
float fBM(vec3 p, float amplitude, float frequency, float lacunarity, float gain, int octaves) {
    float sum = 0.0;

    for (int i = 0; i < octaves; i++) {
        // Add the base noise with the current amplitude
        sum += amplitude * noise(p * frequency);

        // Increase frequency and reduce amplitude for the next octave
        frequency *= lacunarity;
        amplitude *= gain;
    }

    return sum;
}


// Function to perturb the distance of a primitive
// val: the Val object containing the signed distance and cost
// p: the position in space
// amplitude: the amplitude of the perturbation
// frequency: the frequency of the fBM noise
Val Perturb(Val v, vec3 p, float amplitude, float frequency, float lacunarity, float gain, int octaves) {
    // Compute the fBM noise to perturb the signed distance
    float noiseValue = fBM(p, amplitude, frequency, lacunarity, gain, octaves);

    // Apply the noise to the signed distance
    v.v += noiseValue;

    return v;
}


// Operators -------------------------------------------------------------------------------

// Union
// a,b : field function of left and right sub-trees
Val Union(Val a,Val b)
{
  return Val(min(a.v,b.v),a.c+b.c+1);
}


// Intersection
// a,b : field function of left and right sub-trees
Val Intersection(Val a, Val b)
{
    return Val(max(a.v,b.v),a.c+b.c+1);
}

// Primitives -------------------------------------------------------------------------------

// Sphere
// p : point
// c : center of skeleton
// r : radius
Val Sphere(vec3 p,vec3 c,float r)
{
  return Val(length(p-c)-r,1);
}

// Plane 
// p : point
// n : Normal of plane
// o : Point on plane
Val Plane(vec3 p, vec3 n, vec3 o)
{
    
    return Val(dot((p-o),n),1);
}


// Capsule
// p : point
// a, b : endpoint of the segment
// r : radius
Val Capsule(vec3 p,vec3 a, vec3 b,float r)
{
    // Compute the projection of point p on the line segment [a, b]
    vec3 pa = p - a;    // Vector from a to p
    vec3 ba = b - a;    // Vector from a to b (capsule axis)
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0., 1.0);  // Projection factor along the axis
    
    // Compute the distance from point p to the capsule's axis
    vec3 closestPoint = a + h * ba;  // Closest point on the capsule's axis to p
    float distanceToSurface = length(p - closestPoint) - r;  // Distance from point p to the surface of the capsule
    
    return Val(distanceToSurface, 1);  // Return the distance field and cost
}


// Box
// p : point
// a : bottom left corner
// b : top right corner
Val Box(vec3 p, vec3 a, vec3 b) {
    vec3 proj = vec3 (max(a.x - p.x, p.x - b.x), // Max distance to x
                       max(a.y - p.y, p.y - b.y), // Max distance to y
                       max(a.z - p.z, p.z - b.z)); // Max distance to z

    float outsideDistance = length(max(proj, vec3(0.0))); // if outside
    float insideDistance = min(max(proj.x, max(proj.y, proj.z)), 0.0); // if inside
    
    float distance = outsideDistance + insideDistance;
    
    return Val(distance, 1);
}

//Ellispoide 
//p : point 
//c : center
//d : x,y,z deformation along the x,y,z axis 
Val Ellipsoide(vec3 p, vec3 c, vec3 d){
    // Calculate the normalized value based on the ellipsoid equation
    float value = sqrt(
      ( (p.x - c.x)*(p.x - c.x) ) / (d.x*d.x) 
    + ( (p.y - c.y)*(p.y - c.y) ) / (d.y*d.y)  
    + ( (p.z - c.z)*(p.z - c.z) ) / (d.z*d.z)
    );

    // Check if the point is inside or outside the ellipsoid
    if(value <= 1.0){
        // Inside: return the distance to the nearest border (surface)
        // Use a normalized space, so the distance is scaled by the deformation factors
        float distToBorder = (1.0 - value) * min(d.x, min(d.y, d.z)); // Approximate distance
        
        return Val(-distToBorder, 1);
    } else {
        // Outside: return the distance to the surface of the ellipsoid
        float distToBorder = (value - 1.0) * min(d.x, min(d.y, d.z)); // Approximate distance
        
        return Val(distToBorder, 1);
    }
}

//Tore
//p : point
//c : center
//r1 : radius 1
//r2 : radius 2
//radius order does not matter 
Val Tor(vec3 p, vec3 c, float r1, float r2){
    vec3 prime = vec3(p.x - c.x, p.y - c.y, p.z - c.z);
    float rho = sqrt(prime.x * prime.x + prime.y * prime.y);
    float R = max(r1, r2);  // Major radius
    float r = min(r1, r2);  // Minor radius
    float dist_from_major_radius = abs(rho - R);
    float distance_to_surface = sqrt(dist_from_major_radius * dist_from_major_radius + prime.z * prime.z);
    if(distance_to_surface <= r) {
        
        return Val(-(r - distance_to_surface), 1);
    } else {
        
        return Val(distance_to_surface - r, 1);
    }
}


// Cylinder
// p : point
// a, b : endpoints of the axis segment
// r : radius
Val Cylinder(vec3 p, vec3 a, vec3 b, float r) {

    Val caps = Capsule(p, a, b, r); // Capsule
    Val pa = Plane(p, normalize(a-b), a); // Plane through A
    Val pb = Plane(p, normalize(b-a), b); // Plane through A

    // Intersect the capsule with the plans
    return Intersection(Intersection(caps, pb), pa);
}
// Translate 
// p : origin point
// t : translation vectore 
vec3 translate(vec3 p,vec3 t){
    return p + t;
}

// ----------------------------------------------------------------------------------------------






// Potential field of the object
// p: point
// Apply noise to the object
Val object(vec3 p) {
    
    
    // Set the noise & perturbation parameters
    float lacunarity = 2.0;
    float gain = 0.5;
    int octaves = 4;
    float amplitude = 0.5; // Amplitude of the noise
    float frequency = 1.0; // Frequency of the noise
    float noiseValue = noise(p * frequency);

    // Noisy Sphere
    Val noisy_sphere = Sphere(p, vec3(3.0, 0.0, -1.0), 3.0);
    noisy_sphere.v = noisy_sphere.v + noiseValue * amplitude;
    
    // Perturbed Sphere
    Val perturbed_sphere = Sphere(p, vec3(9.0, 0.0, 9.0), 3.0);
    perturbed_sphere = Perturb(perturbed_sphere, p, amplitude, frequency, lacunarity, gain, octaves);
    
    Val v = Union(noisy_sphere, perturbed_sphere);

    v.v = v.v * 0.6; // Remove artifacts
    
    // Other primitives 
    Val plane = Plane(p, vec3(0., 0., 1.), vec3(0.0, 0.0, -4.0));
    Val caps = Capsule(p, vec3(5., 5., 0.), vec3(8., 8., 0.), 1.5);  
    Val box = Box(p, vec3(8., 8., 3.), vec3(12., 12.0, 7.0));
    Val elip = Ellipsoide(p, vec3(0.,-6.,0.), vec3(3.0,3.0,1.0));
    Val tor = Tor(p, vec3(0.,-10.,5.), 1., 2.);
    Val cyl = Cylinder(p,vec3(0.,-15.,0.), vec3(0.,-10.,0.), 3.0);
        
    v = Union(v, plane);
    v = Union(v, caps);
    v = Union(v, box);
    v = Union(v, elip);
    v = Union(v, tor);
    v = Union(v, cyl);
    
    //Translated object 
     vec3 t = vec3(-25.,-5.,2.); // Example translation vector

    v = Union(v, Sphere(p,translate(vec3(3.,0.,-1.), t),3.));
    v = Union(v, Capsule(p, translate(t,vec3(5., 5., 0.)), translate(t,vec3(8., 8., 0.)), 1.5));
    v = Union(v, Box(p, translate(t,vec3(7., 7., 2.)), translate(t,vec3(10., 10.0, 5.0)))); 
    v = Union(v, Ellipsoide(p, translate(t,vec3(0.,-6.,0.)), vec3(3.0,3.0,1.0)));
    v = Union(v, Tor(p, translate(t,vec3(0.,-10.,5.)), 1., 2.));
    v = Union(v, Cylinder(p,translate(t,vec3(0.,-15.,0.)), translate(t, vec3(0.,-10.,0.)), 3.0));


    return v;
}




// Analysis of the scalar field -----------------------------------------------------------------

const int Steps=200;// Number of steps
const float Epsilon=.01;// Marching epsilon

// Object normal
// p : point
vec3 ObjectNormal(vec3 p)
{
  const float eps=.001;
  vec3 n;
  Val val=object(p);
  float v=val.v;
  n.x=object(vec3(p.x+eps,p.y,p.z)).v-v;
  n.y=object(vec3(p.x,p.y+eps,p.z)).v-v;
  n.z=object(vec3(p.x,p.y,p.z+eps)).v-v;
  return normalize(n);
}

// Trace ray using ray marching
// ray : The ray
//   e : Maximum distance
//   h : hit
//   s : Number of steps
//   c : cost
bool SphereTrace(Ray ray,float e,out float t,out int s,out int c)
{
  bool h=false;
  
  // Start at the origin
  t=0.0;
  c=0;
  
  for(int i=0;i<Steps;i++)
  {
    s=i;
    vec3 p=Point(ray,t);
    Val val=object(p);
    float v=val.v;
    c+=val.c;
    // Hit object
    if(v<0.)
    {
      h=true;
      break;
    }
    // Move along ray
    t+=max(Epsilon,v);
    // Escape marched too far away
    if(t>e)
    {
      break;
    }
  }
  return h;
}


// Texture  -------------------------------------------------------------------------------

// Fonction pour créer des strates concentriques radiales déformées
vec3 ConcentricRings(vec3 p, float scale, float deformation, float frequency, float lacunarity, float gain, int octaves) {
    // Calcule la distance radiale depuis le centre
    float dist = length(p.xy) * scale;

    // Ajoute un bruit pour la déformation
    float noise = fBM(p, deformation, frequency, lacunarity, gain, octaves); // Ajustez les paramètres selon vos besoins
    dist += noise;

    // Génère des strates concentriques avec une fonction sinusoïdale
    float rings = sin(dist * 10.0) * 0.5 + 0.7; // Normalise le sinus pour obtenir [0, 1]

    // Mélange les couleurs pour créer un effet de strates
    vec3 color1 = vec3(0.5, 0.3, 0.1); // Couleur de base des strates
    vec3 color2 = vec3(0.9, 0.5, 0.3); // Couleur secondaire pour les strates
    vec3 finalColor = mix(color1, color2, rings);

    return finalColor;
}


// Lighting -------------------------------------------------------------------------------

// Background color
// ray : Ray 
vec3 background(Ray ray)
{
  return mix(vec3(.45,.55,.99),vec3(.65,.69,.99),ray.d.z*.5+.5);
}



// Function to calculate ambient occlusion
// p : Position of the point to evaluate
// n : Normal at the point
// numSamples : Number of samples
// R : Maximum distance for test rays
float AmbientOcclusion(vec3 p, vec3 n, int numSamples, float R) {
    int hits = 0; // Counter for blocked rays

    // Loop through all samples
    for (int i = 0; i < numSamples; i++) {
        // Generate a random direction in the hemisphere centered on p and oriented along n
        vec3 randomDir = Hemisphere(i, n);
        
        // Create the ray
        vec3 rayOrigin = p + Epsilon * n; // Slightly offset to avoid self-intersection
        vec3 rayDir = randomDir;
        Ray delta = Ray(rayOrigin, rayDir);

        // Check for intersection
        float t; // Distance to the intersection
        int steps, cost; // Variables for SphereTrace (not used here)
        bool hit = SphereTrace(delta, R, t, steps, cost);

        // If the ray intersects an object within distance R, increment the counter
        if (hit && t < R) {
            hits += 1;
        }
    }

    // Calculate occlusion: 1 - (fraction of blocked rays)
    float occlusion = 1.0 - float(hits) / float(numSamples);
    return occlusion;
}





// Hard shadowing
// p : Point
// n : Normal
// l : Light direction
float HardShadow(vec3 p, vec3 n, vec3 l)
{
  float t;
  int s;
  int c;
  bool hit = SphereTrace(Ray(p + Epsilon * n, l), 100., t, s, c);
  if (!hit)
  {
    return 1.;
  }
  return 0.;
}



// Function to sample light points along the segment
// a, b : the two endpoints of the light segment
// i : the current sample index
// numLights : the total number of light points
vec3 SampleLightPoint(vec3 a, vec3 b, int i, int numLights) {
    float t = float(i) / float(numLights - 1); // Fraction of the sample along the segment
    return mix(a, b, t); // Linear interpolation between a and b
}



// Soft shadowing
// p : point to shade
// n : normal of the point
// a, b : endpoints of the light segment
// numLights : number of light points along the segment
float SoftShadow(vec3 p, vec3 n, vec3 a, vec3 b, int numLights) {
    float shadow = 0.0;

    // Sample light points
    for (int i = 0; i < numLights; ++i) {
        vec3 lightPoint = SampleLightPoint(a, b, i, numLights); // Current light point
        vec3 lightDir = normalize(lightPoint - p); // Direction of the light

        // Check if the point is in shadow using the SphereTrace method
        float t;
        int s, c;
        bool hit = SphereTrace(Ray(p + Epsilon * n, lightDir), 100.0, t, s, c);

        // Add to shadow if the ray is blocked
        if (!hit) {
            shadow += 1.0;
        }
    }

    // Calculate the average intensity of the shadow
    return shadow / float(numLights);
}


// Shading and lighting
//   p : Point
//   n : Normal at the point
//   eye : Eye direction
//   c_ambiant : Ambient reflection coefficient
//   c_diffusion : Diffuse reflection coefficient
//   c_speculaire : Specular reflection coefficient
//   c_brillance : Shininess coefficient for the specular term
//   lp1 : First light point
//   lp2 : Last light point (null if it's a point light)
//   numLight : number of lights (must be >= 0)
vec3 Shade(vec3 p, vec3 n, Ray eye, vec3 c_ambiant, vec3 c_diffusion, vec3 c_speculaire, float c_brillance, vec3 lp1, vec3 lp2, int numLight)
{
    float shadow;
    vec3 l = normalize(lp1 - p); 

    if (numLight == 0) {
        shadow = 0.0;
        
    } else if (numLight == 1) {
        shadow = HardShadow(p, n, l);
    
    } else {
        shadow = SoftShadow(p, n, lp1, lp2, numLight);
    
    }
    
    // Light direction based on the first point of the segment

    
    // Calculate ambient occlusion
    float ao = AmbientOcclusion(p, n, 64, 5.0); // Example with 64 samples and radius R = 5.0
    // Phong diffuse
    float diff_strength = clamp(dot(n, l), 0., 1.);
    vec3 diffuse = c_brillance *c_diffusion * diff_strength;

    // Phong specular
    vec3 reflect_source = reflect(-l, n);
    float spec_strength = max(dot(reflect_source, -eye.d), 0.0);
    spec_strength = 0.15 * pow(spec_strength, c_brillance);
    vec3 specular = spec_strength * c_speculaire;

    // Phong shading
    vec3 c = ao * c_ambiant + shadow * (diffuse + specular);
    
    // Uncoment to adds the effect of concentric layers
    // vec3 ringsColor = ConcentricRings(p, 1.0, 0.1, 9.0, 7.0, 0.1, 4); 
    // c *= ringsColor;
    
    return c;
}





// Shading according to the number of steps in sphere tracing
// n : Number of steps
vec3 ShadeSteps(int n, int m)
{
  float t = float(n) / float(m);
  return 0.5 + mix(vec3(0.05, 0.05, 0.5), vec3(0.65, 0.39, 0.65), t);
}

// Image
void mainImage(out vec4 color, in vec2 pxy)
{  
  // Convert pixel coordinates
  vec2 pixel = (-iResolution.xy + 2. * pxy) / iResolution.y;

  // Mouse
  vec2 m = iMouse.xy / iResolution.xy;
  
  // Camera
  Ray ray = CreateRay(m, pixel);
  
  // Trace ray
  
  // Hit and number of steps
  float t = 0.0;
  int s = 0;
  int c;
  bool hit = SphereTrace(ray, 100., t, s, c);
  
  // Shade background
  vec3 rgb = background(ray);
  
  if (hit)
  {
    // Position
    vec3 p = Point(ray, t);
    
    // Compute normal
    vec3 n = ObjectNormal(p);
    
    // Ambient coefficient
    vec3 c_ambiant = .25+.25*background(Ray(p,n));   // Ambiant color
    vec3 c_diffusion = vec3(0.80, 0.80, 0.80); // Diffuse coefficient
    vec3 c_speculaire = vec3(1.0, 1.0, 1.0);   // Specular coefficient
    float c_brillance = 0.35;                  // Shininess coefficient

    // Shade object with light
    
    // Soft Shadow : Uncomment these lines to test them with vector light
    //vec3 lp1 = vec3(5.0, 10.0, 25.0);
    //vec3 lp2 = vec3(9.0, 12.0, 27.0); 
    //int numLight = 16;
    
    // Hard Shadow :  Uncomment these lines to test them with a spot light
    vec3 lp1 = vec3(5.0, 10.0, 25.0);
    vec3 lp2;
    int numLight = 1;
    
    rgb = Shade(p, n, ray, c_ambiant, c_diffusion, c_speculaire, c_brillance, lp1, lp2, numLight);
  }
  
  // Uncomment this line to shade the image with false colors representing the number of steps
  // rgb = ShadeSteps(s, Steps);
  
  // Uncomment this line to shade the cost
  // rgb = ShadeSteps(c, 500);
  
  color = vec4(rgb, 1.);
}