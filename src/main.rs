extern crate rkm;
extern crate ndarray;
extern crate csv;
extern crate itertools;

use ndarray::{Array2, Array3};
use std::str::FromStr;
use std::io::*;
use itertools::{Itertools};
use std::ops::{Add,Sub,Div,Mul};
//This just defines a 4d point (x,y,z,t) and its operations. Ignore everything up to the next comment

#[derive(Debug,Copy,Clone)]
struct Point {
    x:f64,
    y:f64,
    z:f64,
    t:f64,
}

impl PartialEq for Point {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x
            && self.y == other.y
            && self.z == other.z
    }
}
impl Sub for &Point {
    type Output = Point;

    fn sub(self, other: &Point) -> Point {
        Point {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
            t: self.t - other.t,
        }
    }
}
impl Add for &Point {
    type Output = Point;

    fn add(self, other: &Point) -> Point {
        Point {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
            t: self.t + other.t,
        }
    }
}
impl Div<f64> for &Point {
    type Output = Point;

    fn div(self, other: f64) -> Point {
        Point {
            x: self.x / other,
            y: self.y / other,
            z: self.z / other,
            t: self.t / other,
        }
    }
}
impl Mul<f64> for &Point {
    type Output = Point;

    fn mul(self, other: f64) -> Point {
        Point {
            x: self.x * other,
            y: self.y * other,
            z: self.z * other,
            t: self.t * other,
        }
    }
}
//You can stop ignoring now

fn main() {
    let dim:usize = read_input("Dimension (2 or 3)");
    let data = read_test_data(dim);
    let cluster_size:usize = read_input("Cluster Size");
    let num_clusters = data.rows()/cluster_size;

    //we are given a ton of n-dimensional points and are told to find the average value in n-space of the function they are generated from on the range the points are in
    //our strategy is to generate a couple (actually a ton) of fake points to make sure that each 'cell' in n-space has at least something in it
    //We will then just average the function's value for each point of the cell and assign that to the cell
    //and finally we will average the values of all of the cells

    let c_map = if num_clusters==1 { vec![0; data.rows()+1] }
    else { rkm::kmeans_lloyd(&data.view(),num_clusters,None).1 };
    //c_map is in the form 2,1,0... where this means the first point it was given is in cluster 2, second in cluster 1, etc.

    let mut data_points:Vec<Point> = Vec::new();//these are the 'real' provided points
    let mut data_extension:Vec<Point> = Vec::new();//this is where we will put our 'fake' generated points
    let mut c_vector:Vec<Vec<Point>> = vec![Vec::new(); num_clusters];// a list of clusters, which are themselves lists of points

    //we want the ceil of each of these numbers to be at least 1, so that our cell array is at least 1x1x1.
    //if we find any real points which are outside of this we will expand in that direction
    let mut x_max:f64 = 0.0;
    let mut y_max:f64 = 0.0;
    let mut z_max:f64 = 0.0;

    for i in 0..data.rows(){
        let row = data.row(i);
        let cn = c_map.get(i).expect("data map failure");
        let q:&mut Vec<Point> = c_vector.get_mut(*cn).unwrap();
        let x = *row.get(0).unwrap();
        let y = *row.get(1).unwrap();
        let z = if dim>2 {*row.get(2).unwrap()} else {0.0};

        x_max = x_max.max(x);
        y_max = y_max.max(y);
        z_max = z_max.max(z);

        let p = Point{ x, y, z, t: if dim>2 {*row.get(3).unwrap()} else {*row.get(2).unwrap()}, };
        data_points.push(p);
        q.push(p);
    }

    //this defines the amount which (dx,dy,dT) will be stretched. Small values represent guessing less, but fill fewer gaps
    const EXTEND_LENGTH:f64 = 1.0;

    //now, for each pair of (non-equal) points in each cluster, we want to make four 'fake' datapoints
    //corresponding to the mean of their x,y,T and to the point which is +- (dx,dy,dT) from either point
    //also, we can calculate the midpoint of any pair to complete gaps between close points
    let func_midpoint: fn((&Point,&Point)) -> Point = |(p1,p2):(&Point,&Point)| &(p1 + p2) / 2.0;
    let func_extend: fn((&Point,&Point)) -> Vec<Point> = |(p1,p2):(&Point,&Point)| {
        let dp = &(p1 - p2) * EXTEND_LENGTH;
        vec![p1 + &dp, p1 - &dp, p2 + &dp, p2 - &dp]
    };

    c_vector.iter().for_each(|cluster| {
        data_extension.extend( cluster.iter().cartesian_product(cluster).filter(|(x,y)| x!=y).map(func_midpoint) );
        data_extension.extend( cluster.iter().cartesian_product(cluster).filter(|(x,y)| x!=y).flat_map(func_extend) );
    });

    println!("{:?}",c_vector.get(0).unwrap());
    println!("{}",data_extension.len());
    println!("{}",data_points.len());

    let mut array = Array3::<(f64,f64)>::default(((x_max+1.0) as usize,(y_max+1.0) as usize,(z_max+1.0) as usize));

    add_valid_points_to_array(&mut array,data_points);
    let t_sum = avg_of_cells(&array);
    println!("real {:?}",t_sum);
    println!("real {:?}",t_sum.0/t_sum.1);

    add_valid_points_to_array(&mut array,data_extension);
    let t_sum = avg_of_cells(&array);
    println!("real+faked {:?}",t_sum);
    println!("real+faked {:?}",t_sum.0/t_sum.1);
}

fn read_test_data(dim:usize) -> Array2<f64> {
    let path = format!("./data challenge 1 {}d.csv", dim);

    let mut data_reader = csv::Reader::from_path(path).unwrap();
    let mut data: Vec<f64> = Vec::new();
    for record in data_reader.records() {
        for field in record.unwrap().iter() {
            let value = f64::from_str(field);
            data.push(value.unwrap());
        }
    }
    Array2::from_shape_vec((data.len() / (dim + 1), (dim + 1)), data).unwrap()
}

fn read_input<T>(prompt:&str) -> T where T:std::str::FromStr, <T as std::str::FromStr>:: Err: std::fmt::Debug {
    print!("{}: ",prompt);
    let _ = stdout().flush();
    let mut s = String::new();
    let _ = stdin().read_line(&mut s);
    s.trim().parse::<T>().expect("failed to read input of the right type")
}

fn add_valid_points_to_array(array:&mut Array3<(f64,f64)>,points:Vec<Point>){
    let mut ignored_points = 0;
    let array_size = array.raw_dim();
    for dp in points {
        if dp.x < 0.0 || dp.y<0.0 || dp.z < 0.0 //if any point is below zero in any dimension
            || dp.x >= (array_size[0] as f64) || dp.y >= (array_size[1] as f64) || dp.z >= (array_size[2] as f64) // or outside the bounds of the real points
        {ignored_points+=1;continue;}//ignore it. this means we aren't adding any new cells
        let (t,amt) = array[[dp.x.floor() as usize, dp.y.floor() as usize, dp.z.floor() as usize]];
        array[[dp.x.floor() as usize, dp.y.floor() as usize, dp.z.floor() as usize]] = (t+dp.t,amt+1.);
    }

    println!("ignored {}",ignored_points);
}

fn avg_of_cells(array:& Array3<(f64,f64)>,) -> (f64,f64){
    let mut empty_cells=0;
    let ret = array.iter().fold((0.0,0.0),|(sum,samt),(t,amt)|
        if amt > &0.0 {
            (sum+(t/amt),samt+1.0)
        }
        else {
            empty_cells+=1;
            (sum,samt)
        });

    println!("There were {} empty cells",empty_cells);
    ret
}