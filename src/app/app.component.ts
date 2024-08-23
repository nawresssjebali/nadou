import { Component } from '@angular/core';
import { RouterModule, RouterOutlet } from '@angular/router';
import { CourierDetailsComponent } from './courier-details/courier-details.component';
import { FirstPageComponent } from './first-page/first-page.component';
import { SchemaComponent } from './schema/schema.component';


@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    RouterOutlet,
    CourierDetailsComponent,
    FirstPageComponent,
    SchemaComponent
   

  ],

  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  title = 'projet-nadou';
}
