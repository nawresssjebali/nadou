import { CommonModule } from '@angular/common';
import { HttpClient, HttpClientModule } from '@angular/common/http';
import { Component, OnInit } from '@angular/core';
import { ChangeDetectorRef } from '@angular/core';


@Component({
  selector: 'app-price',
  standalone: true,
  imports: [CommonModule,HttpClientModule],
  templateUrl: './price.component.html',
  styleUrl: './price.component.css'
})
export class PriceComponent implements OnInit {
 
  
    hubCosts: number = 0;
    routeCosts: number = 0;
    objective: number = 0;
    total_price: number = 0;
    isLoading: boolean = true;
  
    constructor(private http: HttpClient, private changeDetectorRef: ChangeDetectorRef) {}
  
    ngOnInit(): void {
      this.getRouteData();
    }
  
    getRouteData() {
      this.http.post('http://localhost:5000/compute-route', {})
        .subscribe((response: any) => {
          console.log(response);
          this.hubCosts = response.hub_costs;
          this.routeCosts = response.route_costs;
          this.objective = response.objective;
          this.total_price = response.total_price;
          console.log("Hub Costs:", this.hubCosts);
          console.log("Route Costs:", this.routeCosts);
          console.log("Total Price:", this.total_price);
          this.isLoading = false;
          console.log("Is Loading:", this.isLoading);
          this.changeDetectorRef.detectChanges();  // Manually trigger change detection
        }, (error) => {
          console.error('Error:', error);
          this.isLoading = false;
        });
    }
    handleButtonClick() {
      // Add your custom logic here
      console.log('Button clicked!');
    }
    
  }
  